import os
import pandas as pd


def transform_output(predictions, stats, train_time, eval_time, dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, train_time, eval_time (1 row, 5 columns, with header)
    predictions.csv: match_score, tableA_id, tableB_id, etc.
    """

    predictions.to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)

    pd.DataFrame({
        'f1': [stats.f1().item()],
        'precision': [stats.precision().item()],
        'recall': [stats.recall().item()],
        'train_time': [train_time],
        'eval_time': [eval_time],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)

    return None
