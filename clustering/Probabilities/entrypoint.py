import argparse
import pathtype
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from clustering import exact_clusters, unique_mapping_clusters, tune_sim_threshold
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
    ax.hist(x0, bins=bins, label='0', alpha=0.5)
    ax.hist(x1, bins=bins, label='1', alpha=0.5)
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
    parser.add_argument('-d', '--default', action='store_true')
    args = parser.parse_args()

    input_file = args.input.stem
    input_folder = args.input.parent
    if args.output is None:
        output_folder = input_folder
    else:
        output_folder = args.output
    input_data = input_file[len('predictions_'):]
    output_file = os.path.join(output_folder, f"{input_data}_clustering.txt")

    data = pd.read_csv(args.input, encoding_errors='replace')
    data = data.astype(float)
    if args.default:
        threshold = 0.5
        tune_time = 0
        best_settings = unique_mapping_clusters(data, sim_threshold=threshold)
        mean = [best_settings[0], tune_time, best_settings[3]]
        std = [0,0,0]
    else:
        best_settings, threshold, tune_time, mean, std = tune_sim_threshold(data, unique_mapping_clusters, split=True,
                                                          plot_name = os.path.join(output_folder, f"{input_data}_UMC.png"))
    print(mean, std)
    f = open(output_file, 'w')
    print('Unique Mapping Clustering', file=f)
    print(*['F1', 'P', 'R', 'Cluster Time', 'Tune Time', 'Threshold'], file=f, sep='\t')
    print(*(list(best_settings) + [tune_time, threshold]), file=f, sep='\t' )
    print(f"Average F1 +- STD: {mean[0]}\t{std[0]}", file=f)
    print(f"Average Tune Time +- STD: {mean[1]}\t{std[1]}", file=f)
    print(f"Average Cluster Time +- STD: {mean[2]}\t{std[2]}", file=f)
    f.close()
    if args.default:
        threshold = 0.5
        tune_time = 0
        best_settings = exact_clusters(data, sim_threshold=threshold)
        mean = [best_settings[0], tune_time, best_settings[3]]
        std = [0, 0, 0]
    else:
        best_settings, threshold, tune_time, mean, std = tune_sim_threshold(data, exact_clusters, split=True,
                                                             plot_name = os.path.join(output_folder, f"{input_data}_EC.png"))

    f = open(output_file, 'a')
    print('Exact Clustering', file=f)
    print(*['F1', 'P', 'R', 'Cluster Time', 'Tune Time', 'Threshold'], file=f, sep='\t')
    print(*(list(best_settings) + [tune_time, threshold]), file=f, sep='\t')
    print(f"Average F1 +- STD: {mean[0]}\t{std[0]}", file=f)
    print(f"Average Tune Time +- STD: {mean[1]}\t{std[1]}", file=f)
    print(f"Average Cluster Time +- STD: {mean[2]}\t{std[2]}", file=f)
    f.close()

    plot_histogram(data, os.path.join(output_folder, f"{input_data}_histogram.png"))

