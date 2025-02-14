import sys
sys.path.append('fork-zeroer')

import argparse
import pathtype
import os

import py_entitymatching as em
from transform import transform_input, transform_output
from data_loading_helper.feature_extraction import gather_features_and_labels, gather_similarity_features
import numpy as np
import utils
import time

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('-T', '--transitivity', action='store_true',
                    help="whether to enforce transitivity constraint")
parser.add_argument('-f', '--full', action='store_true',
                    help='To perform matching on the full dataset or only on the test set')

args = parser.parse_args()

print("Hi, I'm ZeroER entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

excl_attributes = ['_id', 'ltable_id', 'rtable_id', 'label']

def add_catalog_information(df, tableA, tableB):
    em.set_ltable(df, tableA)
    em.set_rtable(df, tableB)
    em.set_fk_ltable(df, 'ltable_id')
    em.set_fk_rtable(df, 'rtable_id')
    em.set_key(df, '_id')
    
read_prefixes = ['tableA_', 'tableB_']

dataset, tableA, tableB, GT = transform_input(args.input, read_prefixes, args.full)

# if args.full:
#     exp_data = dataset
# else:
#     # if use_full_dataset = False, only uses test part of the dataset
#     exp_data = dataset[1]

exp_data = dataset[0]
test_data = dataset[1]

em.set_key(tableA, 'id')
em.set_key(tableB, 'id')
add_catalog_information(exp_data, tableA, tableB)

id_df = exp_data[["ltable_id", "rtable_id"]]
cand_features = gather_features_and_labels(tableA, tableB, GT, exp_data)
sim_features = gather_similarity_features(cand_features)
print(sim_features)
sim_features_lr = (None,None)
id_dfs = (None, None, None)
if args.transitivity == True:
    id_dfs = (id_df, None, None)


true_labels = cand_features.gold.values
if np.sum(true_labels)==0:
    true_labels = None

cand_features_test = gather_features_and_labels(tableA, tableB, GT, test_data)
sim_features_test = gather_similarity_features(cand_features_test)
true_labels_test = cand_features_test.gold.values

start_time = time.process_time()
y_pred, time_m = utils.run_zeroer(sim_features, sim_features_lr,id_dfs,
                    true_labels ,True,False,args.transitivity, sim_features_test, true_labels_test)
eval_time = time.process_time() - time_m
train_time = time_m - start_time

pred_df = cand_features_test.copy()
pred_df['pred'] = y_pred

transform_output(pred_df, train_time, eval_time, args.output)
