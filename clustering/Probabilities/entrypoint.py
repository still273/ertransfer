import argparse
import pathtype
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from clustering import exact_clusters, unique_mapping_clusters
from sklearn_clusters import kmeans_probability, kmeans_logits

def plot_histogram(data, save_name):
    X = data['prob_class1'].to_numpy()
    y = data['label'].to_numpy()

    x1 = X
    x1 = x1[y == 1]

    x0 = X
    x0 = x0[y == 0]

    bin_width = 0.01
    bins = np.arange(0, 1 + bin_width, bin_width)

    fig, ax = plt.subplots()
    ax.hist(x0, bins=bins, label='0')
    ax.hist(x1, bins=bins, label='1')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

    plt.savefig(save_name+'.png')

def plot_logits(data, save_name):
    X = data[['logit0', 'logit1']].to_numpy()
    y = data['label'].to_numpy()

    x1 = X
    x1 = x1[y == 1]

    x0 = X
    x0 = x0[y == 0]

    fig, ax = plt.subplots()
    ax.scatter(x0[:, 0], x0[:, 1], marker='.', alpha=0.5, label='0')
    ax.scatter(x1[:, 0], x1[:, 1], marker='.', alpha=0.5, label='1')
    ax.legend()
    plt.show()

    plt.savefig(save_name+'.png')


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
    exact_clusters(data)
    unique_mapping_clusters(data)
    kmeans_probability(data, num_clusters=4)
    kmeans_logits(data, num_clusters=2)

    plot_logits(data, 'test_logits')
    plot_histogram(data, 'test_histogram')
