import argparse
import pathtype
import os
import shutil
import random
import time
from collections import namedtuple

from data import PandasDataset
from transform import transform_input, transform_output
from get_features import get_embeddings, get_sent_embeddings
from get_similarity import cos_sim

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sentence_transformers import SentenceTransformer

from transformers import BertTokenizer, BertConfig, BertModel,\
    RobertaTokenizer, RobertaConfig, RobertaModel,\
    DistilBertTokenizer, DistilBertConfig, DistilBertModel, \
    AlbertTokenizer, AlbertConfig, AlbertModel, \
    XLMTokenizer, XLMConfig,  XLMModel, \
    XLNetTokenizer, XLNetConfig,  XLNetModel,\
    AutoTokenizer, AutoModel

from scipy.special import kl_div, rel_entr
from scipy.stats import entropy





def _get_model(model_name):
    if model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained("albert/albert-base-v2")
        config = AlbertConfig.from_pretrained("albert/albert-base-v2")
        model = AlbertModel.from_pretrained("albert/albert-base-v2")
        dim = config.hidden_size
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        config = BertConfig.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', config=config)
        dim = config.hidden_size
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
        config = XLNetConfig.from_pretrained("xlnet/xlnet-base-cased")
        model = XLNetModel.from_pretrained("xlnet/xlnet-base-cased", config=config)
        dim = config.d_model
    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
        config = XLMConfig.from_pretrained("FacebookAI/xlm-mlm-en-2048")
        model = XLMModel.from_pretrained("FacebookAI/xlm-mlm-en-2048", config=config)
        dim = config.emb_dim
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")
        model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        dim = config.hidden_size
    elif model_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        dim = config.dim
    else:
        print('No known model')
        return None
    return model, tokenizer, config, dim

def plot_histogram(X, y, save_name):
    x1 = X
    x1 = x1[y == 1]

    x0 = X
    x0 = x0[y == 0]


    min_value = min(np.amin(x1), np.amin(x0))
    max_value = max(np.amax(x1), np.amax(x0))
    bins = np.linspace(min_value, max_value, 100)

    cmap = mpl.colormaps['inferno_r']
    fig, ax = plt.subplots()
    ax.hist([x0, x1], bins=bins, label=['0', '1'], color=[cmap(0.7), cmap(0.3)], stacked=True)# alpha=0.5)
    #ax.hist(x1, bins=bins, label='1', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlim([-0.2, 1])
    ax.legend()
    plt.show()

    plt.savefig(save_name)

def plot_heatmap(data, labels, save_name):
    #https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#using-the-helper-function-code-style
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='inferno_r')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel('KL-Divergence', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = heatmap.norm(data.max()) / 2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    textcolors = ['black', 'white']
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = heatmap.axes.text(j, i, f"{data[i, j]:.2f}", horizontalalignment="center",
              verticalalignment="center", color=textcolors[int(heatmap.norm(data[i, j]) > threshold)])

    plt.tight_layout()
    plt.savefig(save_name)

def plot_hist_with_smoothing(x, hist, smoothing, save_name):
    fig, ax = plt.subplots()
    cmap = mpl.colormaps['inferno_r']
    if smoothing == 'LS':
        smooth_value = [0.05*hist.sum()/100]*100
        smooth_value = smooth_value / np.sum(hist+smooth_value)
        hist = hist / np.sum(hist+smooth_value)
        ax.bar(x, hist, width=0.01, color=cmap(0.7))
        ax.bar(x, smooth_value, width=0.01, bottom=hist, color=cmap(0.2))
    elif smoothing == 'FM':
        e = 10 ** (-10)
        hist = hist / np.sum(hist)
        smooth_value = np.zeros_like(hist)
        smooth_value[hist<e]=e*10**8
        ax.bar(x, hist, width=0.01, color=cmap(0.7))
        ax.bar(x, smooth_value, width=0.01, color=cmap(0.2))
    plt.savefig(save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('output', type=str, nargs='?', default='/data/output',
                        help='Output directory to store the output')
    parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                        help='The random state used to initialize the algorithms and split dataset')
    parser.add_argument('-lm', '--languagemodel', type=str, nargs='?', default='RoBERTa',
                        help='The language model to use',
                        choices=['BERT', 'RoBERTa', 'DistilBERT', 'XLNet', 'XLM', 'ALBERT'])
    parser.add_argument('-sm', '--smoothing', type=str, nargs='?', default='FM',
                        help='smoothing method to use: FM fixed minimum, LS Laplace Smoothing',
                        choices=['FM', 'LS'])
    parser.add_argument('-ds', '--dataset_split', type=str, nargs='?', default='kj_split',
                        help='which split of datasets to use')

    args = parser.parse_args()
    main_dir = os.path.join(args.output, args.dataset_split)
    os.makedirs(main_dir, exist_ok=True)
    datasets = [ "d2_abt_buy", "d3_amazon_google",
                "d5_imdb_tmdb", "d6_imdb_tvdb", "d7_tmdb_tvdb", "d8_amazon_walmart"]
    labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    sim_distributions = []

    print("Hi, I'm EMTransformer entrypoint!")
    print("Input taken from: ", args.input)
    print("Input directory: ", os.listdir(args.input))
    print("Output directory: ", os.listdir(main_dir))

    # Step 1. Convert input data into the format expected by the method
    print("Method input: ", os.listdir(args.input))
    prefix_1 = 'tableA_'
    prefix_2 = 'tableB_'
    columns_to_join = None

    hyperparameters = namedtuple('hyperparameters', ['lm',  # language Model
                                                    'batch_size',
                                                    'max_len'  # max number of tokens as input for language model
                                                     ])

    hp = hyperparameters(lm='roberta',
                         batch_size=64,
                         max_len=256,
                         )

    for dataset in datasets:
        in_folder = os.path.join(args.input, dataset, args.dataset_split)
        out_folder = os.path.join(main_dir, dataset)
        os.makedirs(out_folder, exist_ok=True)
        train_df, tableA, tableB = transform_input(in_folder, columns_to_join, ' ', [prefix_1, prefix_2])

        #MODEL = "cardiffnlp/twitter-roberta-base-2021-124m"
        #model = AutoModel.from_pretrained(MODEL)
        #tokenizer = AutoTokenizer.from_pretrained(MODEL)

        model = SentenceTransformer("all-mpnet-base-v2")

        datasetA = PandasDataset(tableA, 'AgValue', 'id')
        datasetB = PandasDataset(tableB, 'AgValue', 'id')

        embsA, idsA = get_sent_embeddings(datasetA, model, hp)
        embsB, idsB = get_sent_embeddings(datasetB, model, hp)
        embsA = np.array(embsA)
        embsB = np.array(embsB)

        idsA = pd.Series(data = list(range(len(idsA))), index=idsA, dtype=int)
        idsB = pd.Series(data = list(range(len(idsB))), index=idsB, dtype=int)
        index_A = idsA.loc[train_df['tableA_id']].to_numpy()
        index_B = idsB.loc[train_df['tableB_id']].to_numpy()
        embs_pairA = torch.tensor(embsA[index_A])
        embs_pairB = torch.tensor(embsB[index_B])


        similarity = cos_sim(embs_pairA, embs_pairB)
        similarity=np.array(similarity)
        plot_histogram(similarity, train_df['label'].to_numpy(), os.path.join(out_folder, 'cosine_sim_sent_emb.png'))

        sim_distributions.append(similarity)
    out_file = os.path.join(main_dir, 'KL-Divergence_LS_DA.txt')
    f = open(out_file, 'w')
    print(*['Dataset 1', 'Dataset 2', 'KL Divergence'], sep = '\t', file=f)
    f.close()
    kl_divs = []
    for i, dataset_i in enumerate(datasets):
        kl_divs_i = []
        for j, dataset_j in enumerate(datasets):
            sims_i = sim_distributions[i]
            sims_j = sim_distributions[j]

            min_value = min(np.amin(sims_i), np.amin(sims_j))
            max_value = max(np.amax(sims_i), np.amax(sims_j))
            e = 10 ** (-12)

            dist_i, _ = np.histogram(sims_i, bins=100, range=(min_value, max_value))
            print(f"D1: {dataset_i}, bins in need of smoothing: {np.sum(dist_i<e)}")
            #plot_hist_with_smoothing(np.linspace(min_value, max_value, 100), dist_i, args.smoothing, os.path.join(main_dir, f"{dataset_i}-{dataset_j}_{args.smoothing}.png"))
            if args.smoothing == 'LS':
                dist_i = dist_i + 0.005*dist_i.sum()/100
            dist_i = dist_i / np.sum(dist_i)
            if args.smoothing == 'FM':
                dist_i[dist_i < e] = e

            dist_j, _ = np.histogram(sims_j, bins=100, range=(min_value, max_value))
            print(f"D2: {dataset_j}, bins in need of smoothing: {np.sum(dist_j < e)}")
            print('---------')
            if args.smoothing == 'LS':
                dist_j = dist_j + 0.005*dist_j.sum()/100
            dist_j = dist_j / np.sum(dist_j)
            if args.smoothing == 'FM':
                dist_j[dist_j < e] = e


            kldiv = entropy(dist_i, dist_j)
            kl_divs_i.append(kldiv)
            f = open(out_file, 'a')
            print(*[dataset_i, dataset_j, kldiv], file=f, sep='\t')
            f.close()
        kl_divs.append(kl_divs_i)

    plot_heatmap(np.array(kl_divs), labels, os.path.join(main_dir, f'KL-Divergence_{args.smoothing}.pdf'))


