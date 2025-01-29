import os
import pandas as pd


def transform_output(stats, train_time, eval_time, dest_dir):
    pd.DataFrame({
        'f1': [stats.f1().item()],
        'precision': [stats.precision().item()],
        'recall': [stats.recall().item()],
        'train_time': [train_time],
        'eval_time': [eval_time],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)

    return None
