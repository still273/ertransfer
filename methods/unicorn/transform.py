import os
import sys
import pandas as pd
import torch


def join_columns(table, columns_to_join=None, separator=',', prefixes=['tableA_', 'tableB_'], pair_sep=' [SEP] ',
                 with_instructions=False):
    agg_table = pd.DataFrame()
    for prefix in prefixes:
        if columns_to_join == None:
            columns = [column for column in table.columns if (column != prefix + 'id' and prefix in column)]
        else:
            columns = [prefix + column for column in columns_to_join]

        red_table = table.loc[:, columns]
        red_table = red_table.fillna('')
        red_table = red_table.astype(str)

        part_table = red_table.aggregate(separator.join, axis=1)
        part_table.rename(prefix + 'AgValue', inplace=True)

        agg_table = pd.concat([agg_table, table[prefix + 'id'], part_table], axis=1)
    if with_instructions:
        agg_table['pairs'] = agg_table.apply(lambda row:
                                             f"does {row[prefixes[0]+'AgValue']}{pair_sep}matches with {row[prefixes[1]+'AgValue']}",
                                             axis=1)
    else:
        agg_table['pairs'] = agg_table[[prefixes[0]+'AgValue', prefixes[1]+'AgValue']].aggregate(pair_sep.join, axis=1)
    agg_table = pd.concat([agg_table, table['label']], axis=1)
    return agg_table[[prefixes[0]+'id', prefixes[1]+'id', 'pairs', 'label']]


def transform_input(target_dir, columns_to_join=None, separator=',', prefixes=['tableA_', 'tableB_'],
                    full_test=False, with_instructions=False):

    train_df = pd.read_csv(os.path.join(target_dir, 'train.csv'), encoding_errors='replace')
    valid_df = pd.read_csv(os.path.join(target_dir, 'valid.csv'), encoding_errors='replace')
    test_df = pd.read_csv(os.path.join(target_dir, 'test.csv'), encoding_errors='replace')

    if full_test == 'vt':
        test_df = pd.concat([train_df, valid_df, test_df],ignore_index=True)
    elif full_test == 'v':
        test_df = pd.concat([valid_df, test_df], ignore_index=True)

    target_test = join_columns(test_df, columns_to_join, separator, prefixes, with_instructions=with_instructions)
    print(target_test['pairs'])

    return target_test


def transform_output(predictions, labels, l_id, r_id, results_per_epoch,preprocess_time, train_time, eval_time,
                     test_input,  dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, train_time, eval_time (1 row, 5 columns, with header)
    predictions.csv: tableA_id, tableB_id, etc. (should have at least 2 columns and a header row)
    """
    sm = torch.nn.Softmax(dim=1)
    print(predictions)
    test_name = str(test_input).split('/')[-2]
    probs = sm(torch.Tensor(predictions))
    predictions_df = pd.DataFrame(data={'tableA_id':l_id, 'tableB_id':r_id, 'prob_class1':probs[:,1].tolist(), 'label':labels})

    predictions_df.to_csv(os.path.join(dest_dir, f'predictions_{test_name}.csv'), index=False)
    predictions_df['predictions'] = (predictions_df['prob_class1']>0.5).astype(int)

    if predictions_df['predictions'].sum() > 0:
        # calculate evaluation metrics
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
        'preprocess_time': [preprocess_time],
        'train_time': [train_time],
        'eval_time': [eval_time],
    }).to_csv(os.path.join(dest_dir, f'metrics_{test_name}.csv'), index=False)

    if type(results_per_epoch) != type(None):

        epoch_res_cols = ['epoch', 'train_f1', f'train_f1_{test_name}', f'valid_f1_{test_name}']
        epoch_res_cols += ['train_time', 'test_train_data_time', 'valid_time']

        pd.DataFrame(results_per_epoch,
                     columns=epoch_res_cols
                     ).to_csv(os.path.join(dest_dir, 'metrics_per_epoch.csv'), index=False)

    return None


