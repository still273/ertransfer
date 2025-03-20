import os
import sys

import pandas as pd
import torch


def join_columns(table, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
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
        # part_table = part_table.map(lambda x: x.replace('nan', ''))
        part_table.rename(prefix + 'AgValue', inplace=True)

        agg_table = pd.concat([agg_table, table[prefix + 'id'], part_table], axis=1)

    return pd.concat([agg_table, table['label']], axis=1)


def transform_input(source_dir, add_test_data, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_'],
                    full_train_input=False, full_add_test=False):

    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'), encoding_errors='replace')
    valid_df = pd.read_csv(os.path.join(source_dir, 'valid.csv'), encoding_errors='replace')

    if full_train_input == 'vt':
        test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'), encoding_errors='replace')
        train_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)  # test_df
    elif full_train_input == 'v' :
        test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'), encoding_errors='replace')
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        valid_df = test_df
    print(train_df.shape)
    train = join_columns(train_df, columns_to_join, separator, prefixes)
    valid = join_columns(valid_df, columns_to_join, separator, prefixes)

    test_files = []

    for i, folder in enumerate(add_test_data):
        test_df = pd.read_csv(os.path.join(folder, 'test.csv'), encoding_errors='replace')
        if full_add_test and not str(folder) == str(source_dir):
            train_df = pd.read_csv(os.path.join(folder, 'train.csv'), encoding_errors='replace')
            valid_df = pd.read_csv(os.path.join(folder, 'valid.csv'), encoding_errors='replace')
            test_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        print(test_df.shape)
        test= join_columns(test_df, columns_to_join, separator, prefixes)
        test_files.append(test)
        print(set(test_df['label'].tolist()))

    return train, valid, test_files


def transform_output(predictions, logits, test_table, results_per_epoch,preprocess_time, train_time, eval_time, test_input, train_size, dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, train_time, eval_time (1 row, 5 columns, with header)
    predictions.csv: tableA_id, tableB_id, etc. (should have at least 2 columns and a header row)
    """
    epoch_res_cols = ['epoch']
    sm = torch.nn.Softmax(dim=1)
    for i, test_folder in enumerate(test_input):
        test_name = str(test_folder).split('/')[-2]
        probs = sm(torch.Tensor(logits[i]))
        predictions_df = predictions[i]
        predictions_df.rename(columns={'labels': 'label'}, inplace=True)
        predictions_df['prob_class1'] = probs[:, 1].tolist()
        predictions_df['tableA_id'] = test_table[i].loc[predictions_df.index, 'tableA_id']
        predictions_df['tableB_id'] = test_table[i].loc[predictions_df.index, 'tableB_id']
        predictions_df[['tableA_id', 'tableB_id', 'label', 'prob_class1']].to_csv(os.path.join(dest_dir, f'predictions_{test_name}.csv'),
                                                                                  index=False)

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
            'preprocess_time': [preprocess_time[i]],
            'train_time': [train_time],
            'eval_time': [eval_time[i]],
            'train_size': [train_size],
        }).to_csv(os.path.join(dest_dir, f'metrics_{test_name}.csv'), index=False)
        epoch_res_cols += [f'f1_{test_name}', f'precision_{test_name}', f'recall_{test_name}']
    if type(results_per_epoch) != type(None):
        if len(results_per_epoch[0]) < len(test_input) * 3 + 4:
            epoch_res_cols = epoch_res_cols[:4]
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
