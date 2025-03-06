import argparse
from os import path
import os
import random
import pathtype
import pandas as pd
import numpy as np

def exact_clusters(data, limit=1, sim_threshold=0.7):
    top_matches_tableA = set([])
    top_matches_tableB = set([])

    ids_tableA = set(data['tableA_id'].to_list())
    for id in ids_tableA:
        part_table = data[(data['tableA_id'] == id) & (data['prob_class1'] > sim_threshold)]
        if part_table.shape[0] == 0:
            continue
        part_table = part_table.sort_values(by=['prob_class1'], ascending=False)
        top_matches_tableA = top_matches_tableA.union(set(part_table[['tableA_id', 'tableB_id']].iloc[:limit].itertuples(index=False, name=None)))

    ids_tableB = set(data['tableB_id'].to_list())
    for id in ids_tableB:
        part_table = data[(data['tableB_id'] == id) & (data['prob_class1'] > sim_threshold)]
        if part_table.shape[0] == 0:
            continue
        part_table = part_table.sort_values(by=['prob_class1'], ascending=False)
        top_matches_tableB = top_matches_tableB.union(set(part_table[['tableA_id', 'tableB_id']].iloc[:limit].itertuples(index=False, name=None)))

    top_matches = top_matches_tableA.intersection(top_matches_tableB)
    data['cluster'] = data.apply(lambda row: (row['tableA_id'], row['tableB_id']) in top_matches, axis=1)
    candidates = data[data['cluster'] == True]

    num_candidates = candidates.shape[0]
    TP = candidates['label'].sum()
    GT = data['label'].sum()

    print(f'Cluster: F1 {2 * TP / (num_candidates + GT)}, P {TP / num_candidates}, R {TP / GT}')

def unique_mapping_clusters(data, sim_threshold=0.70):
    data = data.sort_values(by=['prob_class1'], ascending=False)
    top_matches = set([])
    tableA_id = set([])
    tableB_id = set([])
    for row in data.itertuples(index=False):
        if row.prob_class1 < sim_threshold:
            break
        if row.tableA_id not in tableA_id and row.tableB_id not in tableB_id:
            top_matches.add((row.tableA_id, row.tableB_id))
            tableA_id.add(row.tableA_id)
            tableB_id.add(row.tableB_id)

    data['cluster'] = data.apply(lambda row: (row['tableA_id'], row['tableB_id']) in top_matches, axis=1)
    candidates = data[data['cluster'] == True]

    num_candidates = candidates.shape[0]
    TP = candidates['label'].sum()
    GT = data['label'].sum()

    print(f'Cluster: F1 {2 * TP / (num_candidates + GT)}, P {TP / num_candidates}, R {TP / GT}')


