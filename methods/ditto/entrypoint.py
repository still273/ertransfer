import sys
sys.path.append('fork-ditto')

import os
import pathtype
import argparse
import random
import sys
import torch
import numpy as np
from collections import namedtuple
import time

from transform import transform_input, transform_output, save_vectors

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import ProductDKInjector, GeneralDKInjector
from ditto_light.ditto import train
from matcher import classify, load_model

from scipy.special import softmax


parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=str, nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('--test_data', '-t', type=pathtype.Path(readable=True), nargs='*',
                  help='Input directories containing additional test data')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=2,
                    help='Number of epochs to train the model')
parser.add_argument('-if', '--input_train_full', type=str, default=None, nargs='?',
                    choices=['v', 'vt'], help='v: use also valid data for training, validate on test data, vt: use valid and test data for training')
parser.add_argument('-tf', '--test_full', action='store_true',
                    help='Evaluate the full candidates of the additional test data')
parser.add_argument('-pt', '--prev_trained', action='store_true',
                    help='use stored model if available')
parser.add_argument('-le','--last_epoch', action='store_true',
                    help='store model at last epoch')

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
temp_output = os.path.join(args.output, 'temp')
os.makedirs(temp_output, exist_ok=True)

print("Hi, I'm DITTO entrypoint!")
print("Input taken from: ", args.input)
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

test_input = [args.input]
if type(args.test_data) != type(None):
    print(args.test_data)
    test_input += args.test_data

# Step 1. Convert input data into the format expected by the method
print("Method input: ", os.listdir(args.input))
prefix_1 = 'tableA_'
prefix_2 = 'tableB_'
trainset, validset, testsets, train_ids, valid_ids, test_ids = transform_input(args.input, test_input, temp_output,
                                                                               prefixes=[prefix_1, prefix_2],
                                                                               full_train_input=args.input_train_full,
                                                                               full_add_test=args.test_full)

hyperparameters = namedtuple('hyperparameters', ['lm', #language Model
                                                 'n_epochs', #number of epochs
                                                 'batch_size',
                                                 'max_len', #max number of tokens as input for language model
                                                 'lr', #learning rate
                                                 'save_model',
                                                 'last_epoch',
                                                 'logdir',
                                                 'fp16', #train with half precision
                                                 'da', #data augmentation
                                                 'alpha_aug',
                                                 'dk', #domain knowledge
                                                 'summarize', #summarize to max_len
                                                 'size',#dataset size
                                                 'run_id'])

hp = hyperparameters(lm = 'roberta',
                     n_epochs = args.epochs,
                     batch_size = 64,
                     max_len = 256,
                     lr = 3e-5,
                     save_model = args.prev_trained,
                     last_epoch = args.last_epoch,
                     logdir = temp_output,
                     fp16 = True,
                     da = 'all',
                     alpha_aug = 0.8,
                     dk = 'general',
                     summarize = True,
                     size = None,
                     run_id = 0)




#parser.add_argument("--finetuning", dest="finetuning", action="store_true")

# set seeds
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# only a single task for baseline
task = args.input

# create the tag of the run
run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
        hp.dk, hp.summarize, str(hp.size), hp.run_id)
run_tag = run_tag.replace('/', '_')
use_prev_trained = False
t_preprocess = []
if args.prev_trained and os.path.exists(os.path.join(hp.logdir, hp.lm)):
    use_prev_trained = True
    test_datasets = []

    for i, testset in enumerate(testsets):
        t_pstart = time.process_time()
        config = {"task_type": "classification",
                  "vocab": ["0", "1"],
                  "testset": testset}
        if hp.summarize:
            summarizer = Summarizer(config, lm=hp.lm)
            testsets[i] = summarizer.transform_file(testset, max_len=hp.max_len, overwrite=True)

        if hp.dk is not None:
            if hp.dk == 'product':
                injector = ProductDKInjector(config, hp.dk)
            else:
                injector = GeneralDKInjector(config, hp.dk)
            testsets[i] = injector.transform_file(testset, overwrite=True)
        test_datasets += [DittoDataset(testset, lm=hp.lm)]
        t_preprocess += [time.process_time() - t_pstart]
    #test_datasets = [DittoDataset(testset, lm=hp.lm) for testset in testsets]

    matcher = load_model(hp.lm, hp.logdir, hp.lm, torch.cuda.is_available(), hp.fp16)
    threshold = 0.5
    results_per_epoch = None
    train_time = 0
    train_size = 0
else:
    if args.input_train_full:
        config = {"task_type": "classification",
                  "vocab": ["0", "1"],
                  "trainset": trainset}
    else:
        config = {"task_type": "classification",
          "vocab": ["0", "1"],
          "trainset": trainset,
          "validset":validset,
          "testset": testsets[0]}  #idf only build from first testset, not the others.

    # summarize the sequences up to the max sequence length
    t_pstart = time.process_time()
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        for i, testset in enumerate(testsets):
            testsets[i] = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        for i, testset in enumerate(testsets):
            testsets[i] = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    train_size = len(train_dataset)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_datasets = []
    for testset in testsets:
        test_datasets.append(DittoDataset(testset, lm=hp.lm))
    t_preprocess += [time.process_time() - t_pstart]
    # train and evaluate the model
    start_time = time.process_time()
    matcher, threshold, results_per_epoch = train(train_dataset,valid_dataset, [test_datasets[0]], run_tag, hp)
    train_time = time.process_time() - start_time



preds = []
scores = []
labels = []
eval_time = []
for i, testset in enumerate(testsets):
    start_time = time.process_time()
    predictions, logits, label, embs = classify(testset, matcher, lm=hp.lm,
                               batch_size = hp.batch_size,
                               max_len=hp.max_len,
                               threshold=threshold, return_embeddings=True)

    score = softmax(logits, axis=1)
    preds.append(predictions)
    scores.append(logits)
    labels.append(label)

    test_name = str(test_input[i]).split('/')[-2]
    eval_time += [time.process_time() - start_time]
    #save_vectors(embs, label, test_ids[i], os.path.join(args.output, f"embeddings_{test_name}"))



transform_output(scores, threshold, results_per_epoch, test_ids, labels, t_preprocess, train_time, eval_time, test_input, train_size, args.output)

# Step 4. Delete temporary files
for file in os.listdir(temp_output):
    if os.path.isfile(os.path.join(temp_output, file)):
        os.remove(os.path.join(temp_output, file))
