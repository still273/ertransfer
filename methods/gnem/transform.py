import os
import pandas as pd


def transform_output(f1s, ps, rs, runtime, max_mem, dest_dir):
    # save evaluation metrics to metrics.csv
    pd.DataFrame({
        'f1': f1s,
        'precision': ps,
        'recall': rs,
        'max_mem': [max_mem],
        'time': [runtime],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)
    return None
