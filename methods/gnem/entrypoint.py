import sys
sys.path.append('fork-gnem')

import argparse
import pathtype
import pandas as pd

from transform import transform_output
import time
import os
from train_GNEM import train
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import MatchingDataset, collate_fn, MergedMatchingDataset
from torch.utils.data import DataLoader
from EmbedModel import EmbedModel
from GCN import gcn
from logger import set_logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='../../datasets/d2_abt_buy',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='../../output/gnem',
                    help='Output directory to store the output')
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1,
                    help='Number of epochs to train the model')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Hi, I'm DeepMatcher entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

train_table = pd.read_csv(os.path.join(args.input, 'train.csv'), encoding_errors='replace')
test_table = pd.read_csv(os.path.join(args.input, 'test.csv'), encoding_errors='replace')

train_table = train_table.loc[:,['tableA_id', 'tableB_id', 'label']]
print(train_table)
train_table.columns = ['ltable_id', 'rtable_id', 'label']
test_table = test_table.loc[:,['tableA_id', 'tableB_id', 'label']]
test_table.columns = ['ltable_id', 'rtable_id', 'label']
train_table.to_csv(os.path.join(args.output, 'train.csv'), index=False)
test_table.to_csv(os.path.join(args.output, 'test.csv'), index=False)

tableA = pd.read_csv(os.path.join(args.input, 'tableA.csv'), encoding_errors='replace')
str_cols = [col for col in tableA.columns if col != 'id']
tableA[str_cols] = tableA[str_cols].astype(str)
tableB = pd.read_csv(os.path.join(args.input, 'tableB.csv'), encoding_errors='replace')
str_cols = [col for col in tableB.columns if col != 'id']
tableB[str_cols] = tableB[str_cols].astype(str)


useful_field_num = len(tableA.columns)-1
gcn_dim = 768

#val_dataset = MergedMatchingDataset(args.val_path, tableA, tableB, other_path=[args.train_path, args.test_path])
test_dataset = MergedMatchingDataset(os.path.join(args.output, 'test.csv'), tableA, tableB, other_path=[os.path.join(args.output, 'train.csv')])
train_dataset = MatchingDataset(os.path.join(args.output, 'train.csv'), tableA, tableB)

train_iter = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
#val_iter = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
test_iter = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)

embedmodel = EmbedModel(useful_field_num=useful_field_num,device=0)

gcn_layer = 1
dropout = 0.0
model = gcn(dims=[gcn_dim]*(gcn_layer + 1),  dropout=dropout)

no_decay = ['bias', 'LayerNorm.weight']
weight_decay = 0.0
embed_lr = 0.00002
lr = 0.0001

optimizer_grouped_parameters = [
    {'params': [p for n, p in embedmodel.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay, 'lr': embed_lr},
    {'params': [p for n, p in embedmodel.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0, 'lr': embed_lr},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay, 'lr': lr},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0, 'lr': lr}
]

num_train_steps = len(train_iter) * args.epochs
opt = AdamW(optimizer_grouped_parameters, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_train_steps)

model_dir = args.output
log_dir = os.path.join(model_dir, "logs")
tf_log_dir = os.path.join(model_dir, "tf_logs")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(tf_log_dir):
    os.makedirs(tf_log_dir)

logger = set_logger(os.path.join(log_dir, str(time.time()) + ".log"))
tf_logger = SummaryWriter(tf_log_dir)

start_epoch = 0
start_f1 = 0.0

pos_neg_ratio = 1.0

embedmodel = embedmodel.to(embedmodel.device)
model = model.to(embedmodel.device)
pos = 2.0 * pos_neg_ratio / (1.0 + pos_neg_ratio)
neg = 2.0 / (1.0 + pos_neg_ratio)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([neg, pos])).to(embedmodel.device)

start_time = time.process_time()
f1s, ps, rs, score_dicts = train(train_iter, model_dir, logger, tf_logger, model, embedmodel, opt, criterion, args.epochs, test_iter=test_iter,# val_iter=val_iter,
      scheduler=scheduler, log_freq=5, start_epoch=start_epoch, start_f1=start_f1, score_type=['mean'])
train_time = time.process_time() - start_time

transform_output(score_dicts, f1s, ps, rs, train_time, train_time, args.output)

# Step 4. Delete temporary files
os.remove(os.path.join(args.output, 'train.csv'))
os.remove(os.path.join(args.output, 'test.csv'))
