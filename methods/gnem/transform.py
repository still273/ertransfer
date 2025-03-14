import os
import pandas as pd

def transform_input(source_dir, add_test_data, output_dir, prefixes=['tableA_', 'tableB_'],
                    full_train_input=False, full_add_test=False):
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'), encoding_errors='replace')
    valid_df = pd.read_csv(os.path.join(source_dir, 'valid.csv'), encoding_errors='replace')
    if full_train_input:
        test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'), encoding_errors='replace')
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        valid_df = test_df

    train_df = train_df.loc[:, [f'{prefixes[0]}id', f'{prefixes[1]}id', 'label']]
    train_df.columns = ['ltable_id', 'rtable_id', 'label']
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    valid_df = valid_df.loc[:, [f'{prefixes[0]}id', f'{prefixes[1]}id', 'label']]
    valid_df.columns = ['ltable_id', 'rtable_id', 'label']
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)

    test_files = []
    tableAs = []
    tableBs = []
    for i, folder in enumerate(add_test_data):
        test_df = pd.read_csv(os.path.join(folder, 'test.csv'), encoding_errors='replace')
        if full_add_test and not str(folder) == str(source_dir):
            train_df = pd.read_csv(os.path.join(folder, 'train.csv'), encoding_errors='replace')
            valid_df = pd.read_csv(os.path.join(folder, 'valid.csv'), encoding_errors='replace')
            test_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        test_df = test_df.loc[:, [f'{prefixes[0]}id', f'{prefixes[1]}id', 'label']]
        test_df.columns = ['ltable_id', 'rtable_id', 'label']
        test_df.to_csv(os.path.join(output_dir, f'test{i}.csv'), index=False)
        test_files += [f'test{i}.csv']

        tableA = pd.read_csv(os.path.join(folder, 'tableA.csv'), encoding_errors='replace')
        str_cols = [col for col in tableA.columns if col != 'id']
        tableA[str_cols] = tableA[str_cols].astype(str)
        tableB = pd.read_csv(os.path.join(folder, 'tableB.csv'), encoding_errors='replace')
        str_cols = [col for col in tableB.columns if col != 'id']
        tableB[str_cols] = tableB[str_cols].astype(str)
        tableAs += [tableA]
        tableBs += [tableB]
    return tableAs, tableBs, test_files


def transform_output(score_dicts, f1s, ps, rs, preprocess_time,train_time, eval_time, results_per_epoch, test_input, dest_dir):
    epoch_res_cols = ['epoch']
    for i, test_folder in enumerate(test_input):
        test_name = str(test_folder).split('/')[-2]
        # save predictions in predictions.csv
        l_id = []
        r_id = []
        probs = []
        labels = []
        for score_dict in score_dicts[i]:
            for pair in score_dict.keys():
                l_id.append(pair[0])
                r_id.append(pair[1])
                probs.append(score_dict[pair][0]) # see test_GNEM calculate_f1
                labels.append(score_dict[pair][1])

        predictions = pd.DataFrame(data={'tableA_id': l_id, 'tableB_id': r_id, 'label':labels, 'prob_class1':probs})
        predictions = predictions.drop_duplicates()
        predictions.to_csv(os.path.join(dest_dir, f'predictions_{test_name}.csv'), index=False)

        # save evaluation metrics to metrics.csv
        pd.DataFrame({
            'f1': f1s[i],
            'precision': ps[i],
            'recall': rs[i],
            'preprocess_time':[preprocess_time[i]]* len(f1s[i]),
            'train_time': [train_time] * len(f1s[i]),
            'eval_time': [eval_time[i]] * len(f1s[i]),
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
