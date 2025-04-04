import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time

def exact_clusters(data, limit=1, sim_threshold=0.7):
    start_time = time.process_time()
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
    cluster_time = time.process_time() - start_time
    num_candidates = candidates.shape[0]
    TP = candidates['label'].sum()
    GT = data['label'].sum()
    F1 = 2 * TP / (num_candidates + GT)
    P = TP / num_candidates
    R = TP / GT

    #print(f'Cluster: F1 {F1}, P {P}, R {R}')
    return F1, P, R, cluster_time

def unique_mapping_clusters(data, sim_threshold=0.70):
    start_time = time.process_time()
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
    cluster_time = time.process_time() - start_time
    num_candidates = candidates.shape[0]
    TP = candidates['label'].sum()
    GT = data['label'].sum()
    F1 = 2 * TP / (num_candidates + GT)
    P = TP / num_candidates
    R = TP / GT

    #print(f'Cluster: F1 {F1}, P {P}, R {R}')
    return F1, P, R, cluster_time


def tune_sim_threshold(data, cluster_method, min_value=0, max_value=1, min_step=0.01, split=True, plot_name=None, num_runs=10):
    x = np.arange(min_value, max_value, min_step)
    collected_results = []
    full_data = data.copy()
    for run_id in range(num_runs):
        data = full_data
        if split:
            tune, data = train_test_split(data, train_size=0.2, stratify=data['label'])
            y = [list(cluster_method(tune, sim_threshold=sim)) for sim in x]
        else:
            y = [list(cluster_method(data, sim_threshold=sim)) for sim in x]
        y = np.array(y)
        best_f1_idx = np.argmax(y[:,0])
        if split:
            best_settings = list(cluster_method(data, sim_threshold=x[best_f1_idx]))
        else:
            best_settings = y[best_f1_idx]
        collected_results += [[best_settings[0], y[:,3].sum(), best_settings[3]]]
    collected_results = np.array(collected_results)
    print(collected_results)
    mean = np.mean(collected_results, keepdims=True, axis=0)
    std = np.std(collected_results,axis=0, ddof=1, mean=mean)
    print(best_settings, x[best_f1_idx])
    if type(plot_name) != type(None):
        fig, ax = plt.subplots()
        ax.plot(x, y[:,0], '.', label='F1')
        ax.plot(x, y[:,1], '.', label='P')
        ax.plot(x, y[:,2], '.', label='R')
        ax.legend()
        plt.show()

        plt.savefig(plot_name)
    return best_settings, x[best_f1_idx], y[:,3].sum(), mean.flatten(), std


