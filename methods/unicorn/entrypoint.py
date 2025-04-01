import sys
sys.path.append('fork-unicorn')
import argparse
import pathtype
import os
import random
import time
from main import run_unicorn
from main_zero_ins import run_zero_ins
import torch
import pandas as pd

from unicorn.model.encoder import (BertEncoder, MPEncoder, DistilBertEncoder, DistilRobertaEncoder, DebertaBaseEncoder, DebertaLargeEncoder,
                   RobertaEncoder, XLNetEncoder)
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer, DebertaTokenizer, XLNetTokenizer, DistilBertTokenizer
from unicorn.model.matcher import Classifier, MOEClassifier
from unicorn.model.moe import MoEModule
from unicorn.trainer import evaluate
from unicorn.utils.utils import get_data, init_model
from unicorn.dataprocess import predata
from unicorn.utils import param
from transform import transform_input, transform_output

from collections import namedtuple

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=str, nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('--test_data', '-t', type=pathtype.Path(readable=True), nargs='?',
                  help='Input directories containing additional test data')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')
parser.add_argument('-if', '--input_train_full', type=str, default=None, nargs='?',
                    choices=['v', 'vt'], help='v: use also valid data for training, validate on test data, vt: use valid and test data for training')
parser.add_argument('-tf', '--test_full', type=str, default=None, nargs='?',
                    choices=['v', 'vt'], help='Evaluate the full candidates of the additional test data')

parser.add_argument('-lm', '--languagemodel', type=str, nargs='?', default='deberta_base',
                    help='The language model to use')
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=5,
                    help='Number of epochs to train the model')
parser.add_argument('-a', '--adapt', type=str, nargs='?', default='MMD',
                    choices=['MMD', 'IGK'], help='Method for Domain Adaptation')
parser.add_argument('-pt', '--prev_trained', action='store_true',
                    help='use stored model if available')
parser.add_argument('-le','--last_epoch', action='store_true',
                    help='store model at last epoch')
parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/moe/classifier')
parser.add_argument('-z', '--zero', default=False, action='store_true',
                    help='use zero_shot settings with instructions')

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

print("Hi, I'm UNICORN entrypoint!")
print("Input taken from: ", args.input, args.test_data)
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Step 1. Convert input data into the format expected by the method
print("Method input: ", os.listdir(args.input))
prefix_1 = 'tableA_'
prefix_2 = 'tableB_'
columns_to_join = None

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

#
hyperparameters = namedtuple('hyperparameters', ['seed', 'train_seed',
                                                 'model', 'max_seq_length', 'max_grad_norm', 'clip_value', 'batch_size',
                                                 'pre_epochs', 'pre_log_step', 'log_step', 'c_learning_rate', 'num_cls',
                                                 'num_tasks', 'resample', 'modelname', 'ckpt','num_data', 'num_k',
                                                 'scale', 'wmoe', 'expertsnum', 'size_output', 'units', 'shuffle',
                                                 'load_balance', 'balance_loss', 'entroloss', 'pretrain', 'load', 'last_epoch'])

hp = hyperparameters(seed=args.seed, train_seed=args.seed,
                     model = args.languagemodel, max_seq_length=128, max_grad_norm=1.0, clip_value=0.01, batch_size=32,
                     pre_epochs=args.epochs, pre_log_step=10, log_step=10, c_learning_rate=3e-6, num_cls=5,num_tasks=2,
                     resample=0, modelname='UnicornPlus', ckpt='UnicornPlus', num_data=1000, num_k=2, scale=20, wmoe=1,
                     expertsnum=6, size_output=768,units=768, shuffle=1, load_balance=1, balance_loss=0.1, entroloss=0.1,
                     pretrain=args.pretrain, load=False, last_epoch=args.last_epoch)
if args.zero:
    hp = hp._replace(modelname='UnicornZero', ckpt='UnicornZero', shuffle=0, load_balance=0)

if args.pretrain:
    print(hp)
    if args.zero:
        res_per_epoch=run_zero_ins(hp)
    else:
        res_per_epoch = run_unicorn(hp)
    epoch_res_cols = ['epoch', 'avg_valid_f1', 'train_time', 'valid_time']
    pd.DataFrame(res_per_epoch,
                 columns=epoch_res_cols
                 ).to_csv(os.path.join(args.output, 'metrics_per_epoch.csv'), index=False)
else:
    set_seed(hp.train_seed)
    test_df = transform_input(args.input, full_test=args.test_full, with_instructions=args.zero)

    if hp.model in ['roberta', 'distilroberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if hp.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    if hp.model == 'mpnet':
        tokenizer = AutoTokenizer.from_pretrained('all-mpnet-base-v2')
    if hp.model == 'deberta_base':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    if hp.model == 'deberta_large':
        tokenizer = DebertaTokenizer.from_pretrained('deberta-large')
    if hp.model == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    if hp.model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    if hp.model == 'bert':
        encoder = BertEncoder()
    if hp.model == 'mpnet':
        encoder = MPEncoder()
    if hp.model == 'deberta_base':
        encoder = DebertaBaseEncoder()
    if hp.model == 'deberta_large':
        encoder = DebertaLargeEncoder()
    if hp.model == 'xlnet':
        encoder = XLNetEncoder()
    if hp.model == 'distilroberta':
        encoder = DistilRobertaEncoder()
    if hp.model == 'distilbert':
        encoder = DistilBertEncoder()
    if hp.model == 'roberta':
        encoder = RobertaEncoder()

    if hp.wmoe:
        classifiers = MOEClassifier(hp.units)
    else:
        classifiers = Classifier()

    if hp.wmoe:
        exp = hp.expertsnum
        moelayer = MoEModule(hp.size_output, hp.units, exp, load_balance=hp.load_balance)


    encoder = init_model(hp, encoder, restore=hp.ckpt + "_" + param.encoder_path)
    classifiers = init_model(hp, classifiers, restore=hp.ckpt + "_" + param.cls_path)
    if hp.wmoe:
        moelayer = init_model(hp, moelayer, restore=hp.ckpt + "_" + param.moe_path)

    test_metrics = ['f1']
    t_start = time.process_time()
    fea = predata.convert_examples_to_features(test_df[['pairs']].values.tolist(),test_df['label'].tolist(),
                                               hp.max_seq_length, tokenizer,
                                               l_ids=test_df['tableA_id'].tolist(),r_ids=test_df['tableB_id'].to_list())


    test_data_loader = predata.convert_fea_to_tensor(fea, hp.batch_size, do_train=0)
    t_preprocess = time.process_time() - t_start

    t_start = time.process_time()
    predictions, labels, l_ids, r_ids = evaluate.evaluate_moe(encoder, moelayer, classifiers, test_data_loader, args=hp, flag='get_preds')
    t_eval = time.process_time() - t_start

    transform_output(predictions, labels, l_ids, r_ids, None, t_preprocess, 0, t_eval, args.input, args.output)



