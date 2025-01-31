import sys
sys.path.append('fork-deepmatcher')

import argparse
import time
import os
import pathtype

import pandas as pd
import deepmatcher as dm
from HierMatcher import *
from transform import transform_output

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('embedding', type=pathtype.Path(readable=True), nargs='?', default='/workspace/embedding',
                    help='The directory where embeddings are stored')
parser.add_argument('-e', '--epochs', type=int, nargs='?', default=5,
                    help='Number of epochs to train the model')

args = parser.parse_args()

print("Hi, I'm HierMatcher entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

train_data = pd.read_csv(os.path.join(args.input, 'train.csv'), encoding_errors='replace')
test_data = pd.read_csv(os.path.join(args.input, 'test.csv'), encoding_errors='replace')

train_data.drop(columns=['tableA_id', 'tableB_id']).to_csv(os.path.join(args.output, 'train.csv'), index_label='id')
test_data.drop(columns=['tableA_id', 'tableB_id']).to_csv(os.path.join(args.output, 'test.csv'), index_label='id')

# Step 1. Convert input data into the format expected by the method
datasets = dm.data.process(path=args.output,
                           train="train.csv",
                           test="test.csv",
                           id_attr='id',
                           label_attr='label',
                           left_prefix='tableA_',
                           right_prefix='tableB_',
                           cache=None,
                           embeddings_cache_path=args.embedding)

train, test = datasets[0], datasets[1] if len(datasets) >= 2 else None

# Step 2. Run the method
model = HierMatcher(hidden_size=150, embedding_length=300, manualSeed=2)

start_time = time.process_time()
model.run_train(train, test, epochs=args.epochs, batch_size=64, label_smoothing=0.05, pos_weight=1.5)
train_time = time.process_time() - start_time

start_time = time.process_time()
predictions, stats = model.run_eval(test, return_stats=True, return_predictions=True)
eval_time = time.process_time() - start_time

# Step 3. Convert the output into a common format
transform_output(predictions, test_data, stats, train_time, eval_time, args.output)
print("Final output: ", os.listdir(args.output))

# Step 4. Delete temporary files
os.remove(os.path.join(args.output, 'train.csv'))
os.remove(os.path.join(args.output, 'test.csv'))
