import os
import pandas as pd
import torch


def transform_output(predictions, data, stats, train_time, eval_time, dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, train_time, eval_time (1 row, 5 columns, with header)
    predictions.csv: match_score, tableA_id, tableB_id, etc.
    """
    sm = torch.nn.Softmax(dim=1)
    predictions = pd.DataFrame(predictions, columns=['id', 'score1', 'score2'])
    predictions['id'] = predictions['id'].astype(int)
    probs = torch.Tensor(predictions[['score1', 'score2']].to_numpy())
    probs = sm(probs)
    predictions['prob_class1'] = probs[:, 1].tolist()
    #print(predictions)

    #predictions = predictions[predictions['score2']>predictions['score1']]
    predictions['tableA_id'] = data.loc[predictions['id'], 'tableA_id']
    predictions['tableB_id'] = data.loc[predictions['id'], 'tableB_id']
    predictions['label'] = data.loc[predictions['id'], 'label']

    predictions.loc[:,['tableA_id', 'tableB_id', 'label', 'prob_class1']].to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)

    pd.DataFrame({
        'f1': [stats.f1().item()],
        'precision': [stats.precision().item()],
        'recall': [stats.recall().item()],
        'train_time': [train_time],
        'eval_time': [eval_time],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)

    return None
