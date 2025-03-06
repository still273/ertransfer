import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import argparse
import pathtype
from matplotlib import pyplot as plt
import numpy as np
import bz2
import pickle

def load_embeddings_pbz2(name='embeddings'):
    with bz2.BZ2File(name, 'rb') as infile:
        result = pickle.load(infile)

    return np.array(result[0], dtype='float32'), np.array(result[1]), np.array(result[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusters the predicted probabilities')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input File containing the dataset')
    parser.add_argument('output', type=str, nargs='?',
                        help='Output directory to store the output. If not provided, the input directory will be used')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    X, y, pair_ids = load_embeddings_pbz2(args.input)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X_10d = TSNE(n_components=10, method='exact').fit_transform(X)

    clustering = KMeans(n_clusters=2, n_init=10)
    clustering.fit(X_10d)

    preds = clustering.labels_
    labels = y.flatten()
    split0 = labels[preds==0]
    split1 = labels[preds==1]

    print(f'Cluster 0: F1 {2*split0.sum()/(split0.shape[0] +labels.sum())}, P {split0.sum()/split0.shape[0]}, R {split0.sum()/labels.sum()}')
    print(
        f'Cluster 1: F1 {2 * split1.sum() / (split1.shape[0] + labels.sum())}, P {split1.sum() / split1.shape[0]}, R {split1.sum() / labels.sum()}')

    X_2d = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, marker='.', alpha=0.5)
    plt.savefig('scatter_TSNE2d.png')




