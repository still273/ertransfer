import pandas as pd
from sklearn.cluster import KMeans
import argparse
import pathtype
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusters the predicted probabilities')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input File containing the dataset')
    parser.add_argument('output', type=str, nargs='?',
                        help='Output directory to store the output. If not provided, the input directory will be used')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    data = pd.read_csv(args.input, encoding_errors='replace')
    data = data.astype(float)

    clustering = KMeans(n_clusters=2)
    X = data['prob_class1'].to_numpy().reshape(-1,1)
    y = data['label'].to_numpy().reshape(-1,1)

    x1 = X.flatten()
    x1 = x1[y.flatten() == 1]

    x0 = X.flatten()
    x0 = x0[y.flatten() == 0]
    bin_width = 0.01
    bins = np.arange(0, 1+bin_width, bin_width)

    print(x0.shape, x1.shape)


    fig, ax = plt.subplots()
    ax.hist(x0, bins=bins, label='0')
    ax.hist(x1, bins=bins, label='1')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

    plt.savefig('scatter_test.png')

    clustering.fit(data['prob_class1'].to_numpy().reshape(-1, 1))
    preds = clustering.labels_
    labels = y.flatten()
    split0 = labels[preds==0]
    split1 = labels[preds==1]

    print(f'Cluster 0: F1 {2*split0.sum()/(split0.shape[0] +labels.sum())}, P {split0.sum()/split0.shape[0]}, R {split0.sum()/labels.sum()}')
    print(
        f'Cluster 1: F1 {2 * split1.sum() / (split1.shape[0] + labels.sum())}, P {split1.sum() / split1.shape[0]}, R {split1.sum() / labels.sum()}')




