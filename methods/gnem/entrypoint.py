import sys
sys.path.append('fork-gnem')

import argparse
import pathtype
import pandas as pd

from transform import transform_input, transform_output
import time
import os
from train_GNEM import train
from test_GNEM import test
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import MatchingDataset, collate_fn, MergedMatchingDataset
from torch.utils.data import DataLoader
from EmbedModel import EmbedModel
from GCN import gcn
from logger import set_logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import random
import numpy as np

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='../../datasets/d2_abt_buy',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=str, nargs='?', default='../../output/gnem',
                    help='Output directory to store the output')
parser.add_argument('--test_data', '-t', type=pathtype.Path(readable=True), nargs='*',
                  help='Input directories containing additional test data')
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1,
                    help='Number of epochs to train the model')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')
parser.add_argument('-if', '--input_train_full', type=str, default=None, nargs='?',
                    choices=['v', 'vt'], help='v: use also valid data for training, validate on test data, vt: use valid and test data for training')
parser.add_argument('-tf', '--test_full', action='store_true',
                    help='Evaluate the full candidates of the additional test data')
parser.add_argument('-pt', '--prev_trained', action='store_true',
                    help='use stored model if available')
parser.add_argument('-lm', '--languagemodel', type=str, nargs='?', default='BERT',
                    help='The language model to use', choices=['BERT', 'RoBERTa', 'DistilBERT', 'XLNet', 'XLM', 'ALBERT'])
parser.add_argument('-le','--last_epoch', action='store_true',
                    help='store model at last epoch')

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

print("Hi, I'm GNEM entrypoint!")
print("Input taken from: ", args.input)
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

test_input = [args.input]
if type(args.test_data) != type(None):
    print(args.test_data)
    test_input += args.test_data

tableAs, tableBs, test_files = transform_input(args.input, test_input, args.output, full_train_input=args.input_train_full,
                                             full_add_test=args.test_full)


useful_field_num = len(tableAs[0].columns)-1
gcn_dim = 768
model_name = args.languagemodel.lower()

embedmodel = EmbedModel(useful_field_num=useful_field_num, lm = model_name, device=device)

gcn_layer = 1
dropout = 0.0
model = gcn(dims=[gcn_dim]*(gcn_layer + 1),  dropout=dropout)
batch_size = 2

log_dir = os.path.join(args.output, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = set_logger(os.path.join(log_dir, str(time.time()) + ".log"))

pos_neg_ratio = 1.0
pos = 2.0 * pos_neg_ratio / (1.0 + pos_neg_ratio)
neg = 2.0 / (1.0 + pos_neg_ratio)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([neg, pos])).to(embedmodel.device)

t_preprocess = []
if args.prev_trained and os.path.exists(os.path.join(args.output, model_name)):
    checkpoint = torch.load(os.path.join(args.output, model_name))
    embedmodel.load_state_dict(checkpoint["embed_model"])
    model.load_state_dict(checkpoint["model"])
    embedmodel.to(embedmodel.device)
    model.to(embedmodel.device)

    test_iters = []
    for i in range(len(test_input)):
        t_pstart = time.process_time()
        test_dataset = MergedMatchingDataset(os.path.join(args.output, f'test{i}.csv'), tableAs[i], tableBs[i])
        test_iters += [DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)]
        t_preprocess += [time.process_time() - t_pstart]
    # test_datasets = [MergedMatchingDataset(os.path.join(args.output, f'test{i}.csv'), tableAs[i], tableBs[i])
    #                  for i in range(len(test_input))]
    #
    # test_iters = [DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    #               for test_dataset in test_datasets]

    f1s, ps, rs, score_dicts, eval_time = [], [], [], [], []

    for test_i in test_iters:
        t_start = time.process_time()
        test_f1s, test_ps, test_rs, scores = test(iter=test_i, logger=logger, model=model, embed_model=embedmodel, prefix='Test',
                                        crit=criterion, log_freq=len(test_i) // 10, test_step=0, score_type=['mean'])
        f1s += [test_f1s]
        ps += [test_ps]
        rs += [test_rs]
        score_dicts += [scores]
        eval_time += [time.process_time() - t_start]
    train_time = 0
    train_size = 0
    res_per_epoch = None
else:

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
    t_pstart = time.process_time()
    train_dataset = MatchingDataset(os.path.join(args.output, 'train.csv'), tableAs[0], tableBs[0])
    val_dataset = MergedMatchingDataset(os.path.join(args.output, 'valid.csv'), tableAs[0], tableBs[0],
                                        other_path=[os.path.join(args.output, 'train.csv'),
                                                    os.path.join(args.output, 'test0.csv')])
    test_datasets = [MergedMatchingDataset(os.path.join(args.output, f'test{i}.csv'), tableAs[i], tableBs[i])
                     for i in range(len(test_input))]

    batch_size = 2
    train_iter = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_iters = [DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
                  for test_dataset in test_datasets]
    train_size = len(train_dataset)
    t_preprocess += [time.process_time() - t_pstart]

    num_train_steps = len(train_iter) * args.epochs
    opt = AdamW(optimizer_grouped_parameters, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_train_steps)

    tf_logger = SummaryWriter(log_dir)

    start_epoch = 0
    start_f1 = 0.0



    embedmodel = embedmodel.to(embedmodel.device)
    model = model.to(embedmodel.device)


    log_freq = len(train_iter)//10

    start_time = time.process_time()
    f1s, ps, rs, score_dicts, time_m, res_per_epoch = train(train_iter, args.output, logger, tf_logger, model, embedmodel, opt, criterion, args.epochs,
                                                            test_iter=test_iters, val_iter=val_iter,
          scheduler=scheduler, log_freq=log_freq, start_epoch=start_epoch, start_f1=start_f1, score_type=['mean'], save_name=model_name, save_last_epoch=args.last_epoch)
    eval_time = [time.process_time() - time_m]
    train_time =  time_m - start_time

transform_output(score_dicts, f1s, ps, rs, t_preprocess, train_time, eval_time, res_per_epoch, test_input, train_size, args.output)
print("Final output: ", os.listdir(args.output))
