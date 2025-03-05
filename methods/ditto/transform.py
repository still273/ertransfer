import os
import sys

import pandas as pd
import numpy as np
from itertools import product

def join_columns (table, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    agg_table = pd.DataFrame()
    for prefix in prefixes:

        if columns_to_join == None:
            columns = [column for column in table.columns if (column != prefix+'id' and prefix in column)]
        else:
            columns = [prefix+column for column in columns_to_join]

        
        red_table = table.loc[:,columns]
        red_table = red_table.fillna('')
        red_table = red_table.astype(str)
        
        for column in columns:
            red_table[column] = f"COL {column.replace(prefix, '')} VAL " + red_table[column] 
        
        part_table = red_table.aggregate(separator.join, axis=1)
        #part_table = part_table.map(lambda x: x.replace('nan', ''))
        part_table.rename(prefix+'AgValue', inplace=True)
        
        agg_table = pd.concat([agg_table,part_table], axis=1)
    
    return pd.concat([agg_table, table['label']], axis=1),\
        pd.concat([table[prefixes[0] + 'id'], table[prefixes[1] + 'id']], axis=1)


def transform_input(source_dir, add_test_data, output_dir, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'), encoding_errors='replace')
    valid_df = pd.read_csv(os.path.join(source_dir, 'valid.csv'), encoding_errors='replace')
    
    train, train_id = join_columns(train_df, columns_to_join, separator, prefixes)
    valid, valid_id = join_columns(valid_df, columns_to_join, separator, prefixes)
    
    train_file = os.path.join(output_dir, 'train.txt')
    valid_file = os.path.join(output_dir, 'valid.txt')

    train.to_csv(train_file, '\t', header=False, index=False)
    valid.to_csv(valid_file, '\t', header=False, index=False)

    test_files = []
    test_ids = []
    for i, folder in enumerate(add_test_data):
        test_df = pd.read_csv(os.path.join(folder, 'test.csv'), encoding_errors='replace')
        test, test_id = join_columns(test_df, columns_to_join, separator, prefixes)
        test_file = os.path.join(output_dir, f'test{i}.txt')
        test.to_csv(test_file, '\t', header=False, index=False)
        test_files.append(test_file)
        test_ids.append(test_id)

    return train_file, valid_file, test_files, train_id, valid_id, test_ids



def transform_output(scores, threshold, results_per_epoch, ids, labels, train_time, eval_time, test_input, dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, train_time, eval_time (1 row, 5 columns, with header)
    predictions.csv: tableA_id, tableB_id, etc. (should have at least 2 columns and a header row)
    """
    epoch_res_cols = ['epoch']
    for i, test_folder in enumerate(test_input):
        test_name = str(test_folder).split('/')[-2]

        # get the actual candidates (entity pairs with prediction 1)
        predictions_df = pd.DataFrame({'tableA_id':ids[i]['tableA_id'], 'tableB_id':ids[i]['tableB_id'], 'label':labels[i], 'prob_class1':scores[i][:,1]})

        # save candidate pair IDs to predictions.csv
        predictions_df.to_csv(os.path.join(dest_dir, f'predictions_{test_name}.csv'), index=False)

        # calculate evaluation metrics
        predictions_df['predictions'] = (predictions_df['prob_class1'] > threshold).astype(int)
        if predictions_df['predictions'].sum() > 0:
            num_candidates = predictions_df['predictions'].sum()
            true_positives = predictions_df.loc[predictions_df['predictions'] == 1, 'label'].sum()
            ground_truth = predictions_df['label'].sum()
            recall = true_positives / ground_truth
            precision = true_positives / num_candidates
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            precision = 0
            recall = 0

        pd.DataFrame({
            'f1': [f1],
            'precision': [precision],
            'recall': [recall],
            'train_time': [train_time],
            'eval_time': [eval_time],
        }).to_csv(os.path.join(dest_dir, f'metrics_{test_name}.csv'), index=False)

        epoch_res_cols += [f'f1_{test_name}', f'precision_{test_name}', f'recall_{test_name}']
    epoch_res_cols += ['train_time', 'valid_time', 'test_time']
    pd.DataFrame(results_per_epoch,
                 columns=epoch_res_cols
                 ).to_csv(os.path.join(dest_dir, 'metrics_per_epoch.csv'), index=False)
    return None


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    transform_input(in_path)
