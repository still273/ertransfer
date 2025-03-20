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

    fig, ax = plt.subplots()
    ax.hist(x0, bins=bins, label='0', alpha=0.5)
    ax.hist(x1, bins=bins, label='1', alpha=0.5)
    ax.set_yscale('log')
    ax.legend()
    plt.show()

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

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    datasets = ["d1_fodors_zagats",  "d2_abt_buy", "d3_amazon_google", "d4_dblp_acm",
                "d5_imdb_tmdb", "d6_imdb_tvdb", "d7_tmdb_tvdb", "d8_amazon_walmart",
                "d9_dblp_scholar", "d10_imdb_dbpedia"]
    sim_distributions = []

    print("Hi, I'm EMTransformer entrypoint!")
    print("Input taken from: ", args.input)
    print("Input directory: ", os.listdir(args.input))
    print("Output directory: ", os.listdir(args.output))

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
        in_folder = os.path.join(args.input, dataset, 'kj_split')
        out_folder = os.path.join(args.output, dataset)
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
    out_file = os.path.join(args.output, 'KL-Divergence_LS_DA.txt')
    f = open(out_file, 'w')
    print(*['Dataset 1', 'Dataset 2', 'KL Divergence'], sep = '\t', file=f)
    f.close()
    for i, dataset_i in enumerate(datasets):
        for j, dataset_j in enumerate(datasets):
            sims_i = sim_distributions[i]
            sims_j = sim_distributions[j]

            min_value = min(np.amin(sims_i), np.amin(sims_j))
            max_value = max(np.amax(sims_i), np.amax(sims_j))
            e = 10 ** (-10)

            dist_i, _ = np.histogram(sims_i, bins=100, range=(min_value, max_value))
            dist_i = dist_i + 0.05*dist_i.sum()/100
            dist_i = dist_i / np.sum(dist_i)
            #dist_i[dist_i < e] = e

            dist_j, _ = np.histogram(sims_j, bins=100, range=(min_value, max_value))
            dist_j = dist_j + 0.05*dist_j.sum()/100
            dist_j = dist_j / np.sum(dist_j)
            #dist_j[dist_j < e] = e


            kldiv = entropy(dist_i, dist_j)
            f = open(out_file, 'a')
            print(*[dataset_i, dataset_j, kldiv], file=f, sep='\t')
            f.close()



