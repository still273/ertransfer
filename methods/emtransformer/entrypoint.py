import argparse
import pathtype
import os
import shutil
import random
import time

from config import Config
from data_loader import load_data, DataType
from data_representation import InputExample
from optimizer import build_optimizer
from prediction import predict
from torch_initializer import initialize_gpu_seed
from training import train
from transform import transform_input, transform_output
import torch

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('-r', '--recall', type=float, nargs='?', default=0.8,
                    help='Recall value used to select ground truth pairs')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')

parser.add_argument('-m', '--model', default='bert', type=str)  # roberta
parser.add_argument('--max_seq_length', default=128, type=int)  # 180
parser.add_argument('--train_batch_size', default=8, type=int)  # 16
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=5,  # 15.0
                    help='Number of epochs to train the model')

args = parser.parse_args()

print("Hi, I'm EMTransformer entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Step 1. Convert input data into the format expected by the method
print("Method input: ", os.listdir(args.input))
prefix_1 = 'tableA_'
prefix_2 = 'tableB_'
columns_to_join = None
test_df, train_df = transform_input(args.input, columns_to_join, ' ', [prefix_1, prefix_2])
print(test_df.columns, train_df.columns)

device, n_gpu = initialize_gpu_seed(args.seed)
#device, n_gpu = torch.device("cpu"), 0

label_list = [0, 1]
print("training with {} labels: {}".format(len(label_list), label_list))

# Step 2. Run the method
config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model]
if args.model == 'bert':
    model_path = "textattack/bert-base-uncased-yelp-polarity"
elif args.model == 'distilbert':
    model_path = "distilbert-base-uncased"
elif args.model == 'roberta':
    model_path = "cardiffnlp/twitter-roberta-base-emotion"  # "roberta-base"
elif args.model == 'xlnet':
    model_path = "xlnet/xlnet-base-cased"  # "xlnet-base-cased"
elif args.model == 'xlm':
    model_path = "FacebookAI/xlm-mlm-en-2048"
elif args.model == 'albert':
    model_path = "textattack/albert-base-v2-imdb"

if config_class is not None:
    config = config_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=True)
    model = model_class.from_pretrained(model_path, config=config)
    model.to(device)
else:  # SBERT Models
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    model.to(device)

print("initialized {}-model".format(args.model))


train_examples = [InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for
                  i, row in train_df.iterrows()]

training_data_loader = load_data(train_examples,
                                 label_list,
                                 tokenizer,
                                 args.max_seq_length,
                                 args.train_batch_size,
                                 DataType.TRAINING, args.model)

num_train_steps = len(training_data_loader) * args.epochs

optimizer, scheduler = build_optimizer(model,
                                       num_train_steps,
                                       2e-5,
                                       1e-8,
                                       0,
                                       0.0)

start_time = time.process_time()
train(device,
      training_data_loader,
      model,
      optimizer,
      scheduler,
      None,
      args.epochs,
      1.0,
      False,
      experiment_name=args.model,
      output_dir=args.output,
      model_type=args.model)
train_time = time.process_time() - start_time

# Testing
test_examples = [InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for i, row
                 in test_df.iterrows()]

test_data_loader = load_data(test_examples,
                             label_list,
                             tokenizer,
                             args.max_seq_length,
                             args.train_batch_size,
                             DataType.TEST, args.model)

include_token_type_ids = False
if args.model == 'bert':
    include_token_type_ids = True

start_time = time.process_time()
simple_accuracy, f1, classification_report, prfs, predictions, logits = predict(model, device, test_data_loader, include_token_type_ids)
eval_time = time.process_time() - start_time

keys = ['precision', 'recall', 'fbeta_score', 'support']
prfs = {f'class_{no}': {key: float(prfs[nok][no]) for nok, key in enumerate(keys)} for no in range(2)}

print(classification_report)
print(logits)

# Step 3. Convert the output into a common format
transform_output(predictions, logits, test_df, train_time, eval_time, args.output)
print("Final output: ", os.listdir(args.output))
