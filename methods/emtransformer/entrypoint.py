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
from model import save_model
import torch

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=str, nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('--test_data', '-t', type=pathtype.Path(readable=True), nargs='*',
                  help='Input directories containing additional test data')
parser.add_argument('-r', '--recall', type=float, nargs='?', default=0.8,
                    help='Recall value used to select ground truth pairs')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')
parser.add_argument('-if', '--input_train_full', action='store_true',
                    help='Use also the test data of the input for training')
parser.add_argument('-tf', '--test_full', action='store_true',
                    help='Evaluate the full candidates of the additional test data')

parser.add_argument('-lm', '--languagemodel', type=str, nargs='?', default='RoBERTa',
                    help='The language model to use', choices=['BERT', 'RoBERTa', 'DistilBERT', 'XLNet', 'XLM', 'ALBERT'])
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=5,  # 15.0
                    help='Number of epochs to train the model')
parser.add_argument('-pt', '--prev_trained', action='store_true',
                    help='use stored model if available')

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

print("Hi, I'm EMTransformer entrypoint!")
print("Input taken from: ", args.input)
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Step 1. Convert input data into the format expected by the method
print("Method input: ", os.listdir(args.input))
prefix_1 = 'tableA_'
prefix_2 = 'tableB_'
columns_to_join = None

test_input = [args.input]
if type(args.test_data) != type(None):
    print(args.test_data)
    test_input += args.test_data
    test_input = list(dict.fromkeys(test_input))
    print(test_input)

train_df, valid_df, test_dfs = transform_input(args.input, test_input, columns_to_join, ' ', [prefix_1, prefix_2],
                                               args.input_train_full, args.test_full)
print(train_df.columns)

device, n_gpu = initialize_gpu_seed(args.seed)
#device, n_gpu = torch.device("cpu"), 0

label_list = [0, 1]
print("training with {} labels: {}".format(len(label_list), label_list))

# Step 2. Run the method
model_name = args.languagemodel.lower()
max_seq_length = 128
train_batch_size = 16


config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_name]
loaded_model=False
if args.prev_trained == True and os.path.exists(os.path.join(args.output, model_name)):
    model_path = os.path.join(args.output, model_name)
    config_class = None
    loaded_model = True
else:
    if model_name == 'bert':
        model_path = "textattack/bert-base-uncased-yelp-polarity"
    elif model_name == 'distilbert':
        model_path = "distilbert-base-uncased"
    elif model_name == 'roberta':
        model_path = "cardiffnlp/twitter-roberta-base-emotion"  # "roberta-base"
    elif model_name == 'xlnet':
        model_path = "xlnet/xlnet-base-cased"  # "xlnet-base-cased"
    elif model_name == 'xlm':
        model_path = "FacebookAI/xlm-mlm-en-2048"
    elif model_name == 'albert':
        model_path = "textattack/albert-base-v2-imdb"

if config_class is not None:
    config = config_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=True)
    model = model_class.from_pretrained(model_path, config=config)
    model.to(device)
else:  # SBERT Models / self-trained models
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    model.to(device)

print("initialized {}-model".format(model_name))

t_preprocess = []
if not loaded_model:
    t_pstart = time.process_time()
    train_examples = [InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for
                      i, row in train_df.iterrows()]

    training_data_loader = load_data(train_examples,
                                     label_list,
                                     tokenizer,
                                     max_seq_length,
                                     train_batch_size,
                                     DataType.TRAINING, model_name)

    valid_examples = [InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for i, row
                     in valid_df.iterrows()]

    valid_data_loader = load_data(valid_examples,
                                 label_list,
                                 tokenizer,
                                 max_seq_length,
                                 train_batch_size,
                                 DataType.EVALUATION, model_name)

    test_examples = [[InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for i, row
                     in test_df.iterrows()] for test_df in test_dfs]

    test_data_loaders = [load_data(test_example,
                                 label_list,
                                 tokenizer,
                                 max_seq_length,
                                 train_batch_size,
                                 DataType.TEST, model_name) for test_example in test_examples]
    t_preprocess += [time.process_time() - t_pstart]
    num_train_steps = len(training_data_loader) * args.epochs

    optimizer, scheduler = build_optimizer(model,
                                           num_train_steps,
                                           2e-5,
                                           1e-8,
                                           0,
                                           0.0)

    start_time = time.process_time()
    results_per_epoch = train(device,
                              training_data_loader,
                              valid_data_loader,
                              test_data_loaders,
                              model,
                              optimizer,
                              scheduler,
                              args.epochs,
                              1.0,
                              args.prev_trained,
                              experiment_name=model_name,
                              output_dir=args.output,
                              model_type=model_name,
                              tokenizer=tokenizer)
    train_time = time.process_time() - start_time

    # if args.prev_trained:
    #     save_model(model, model_name, args.output, epoch=None, tokenizer=tokenizer)

else:
    test_data_loaders = []
    for test_df in test_dfs:
        t_pstart = time.process_time()
        test_example = [InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for i, row
                      in test_df.iterrows()]
        test_data_loaders += [load_data(test_example,
                                   label_list,
                                   tokenizer,
                                   max_seq_length,
                                   train_batch_size,
                                   DataType.TEST, model_name)]
        t_preprocess += [time.process_time() - t_pstart]
    # test_examples = [[InputExample(i, row[prefix_1 + 'AgValue'], row[prefix_2 + 'AgValue'], row['label']) for i, row
    #                   in test_df.iterrows()] for test_df in test_dfs]
    #
    # test_data_loaders = [load_data(test_example,
    #                                label_list,
    #                                tokenizer,
    #                                max_seq_length,
    #                                train_batch_size,
    #                                DataType.TEST, model_name) for test_example in test_examples]
    results_per_epoch = None
    train_time = 0

include_token_type_ids = False
if model_name == 'bert':
    include_token_type_ids = True


preds = []
logs = []
eval_time = []
for test_data_loader in test_data_loaders:
    start_time = time.process_time()
    simple_accuracy, f1, classification_report, prfs, predictions, logits = predict(model, device, test_data_loader, include_token_type_ids)
    preds += [predictions]
    logs += [logits]
    eval_time += [time.process_time() - start_time]


# Step 3. Convert the output into a common format
transform_output(preds, logs, test_dfs, results_per_epoch,t_preprocess, train_time, eval_time, test_input,args.output)
print("Final output: ", os.listdir(args.output))
