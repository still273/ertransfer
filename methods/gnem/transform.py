import os
import pandas as pd


def transform_output(score_dicts, f1s, ps, rs, runtime, max_mem, dest_dir):

    # save predictions in predictions.csv
    l_id = []
    r_id = []
    for score_dict in score_dicts:
        for pair in score_dict.keys():
            if score_dict[pair][0] > 0.5:  # see test_GNEM calculate_f1
                l_id.append(pair[0])
                r_id.append(pair[1])

    predictions = pd.DataFrame(data={'tableA_id': l_id, 'tableB_id': r_id})
    predictions = predictions.drop_duplicates()
    predictions.to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)

    # save evaluation metrics to metrics.csv
    pd.DataFrame({
        'f1': f1s,
        'precision': ps,
        'recall': rs,
        'max_mem': [max_mem]*len(f1s),
        'time': [runtime]*len(f1s),
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)
    return None
