import sys


sys.path.append('fork-dader')
import argparse
import pathtype
import os
import random
import time
import param
from train.adapt_mmd import train as train_mmd
from train.adapt_mmd import evaluate as evaluate_mmd
from train.evaluate import evaluate as evaluate_gen
from train.pretrain import pretrain,pretrain_best
from train.adapt_invgan_kd import adapt,adapt_best
from modules.extractor import BertEncoder
from modules.matcher import BertClassifier
from modules.alignment import Discriminator
from utils import CSV2Array, convert_examples_to_features, get_data_loader, init_model
from transformers import BertTokenizer
from transform import transform_input, transform_output
import torch
from collections import namedtuple

def convert_table_to_features(table, hp, tokenizer):
    values = table['pairs'].values.tolist()
    labels = table['label'].values.tolist()
    lid = table['tableA_id'].values.tolist()
    rid = table['tableB_id'].values.tolist()
    features = convert_examples_to_features(values, labels,lid, rid, hp.max_seq_length, tokenizer)
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('output', type=str, nargs='?', default='/data/output',
                        help='Output directory to store the output')
    parser.add_argument('--test_data', '-t', type=pathtype.Path(readable=True), nargs='?',
                      help='Input directories containing additional test data')
    parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                        help='The random state used to initialize the algorithms and split dataset')
    parser.add_argument('-if', '--input_train_full', type=str, default=None, nargs='?',
                        choices=['v', 'vt'], help='v: use also valid data for training, validate on test data, vt: use valid and test data for training')
    parser.add_argument('-tf', '--test_full', type=str, default=None, nargs='?',
                        choices=['v', 'vt'], help='Evaluate the full candidates of the additional test data')

    parser.add_argument('-lm', '--languagemodel', type=str, nargs='?', default='BERT',
                        help='The language model to use', choices=['BERT', 'RoBERTa', 'DistilBERT', 'XLNet', 'XLM', 'ALBERT'])
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=5,
                        help='Number of epochs to train the model')
    parser.add_argument('-a', '--adapt', type=str, nargs='?', default='MMD',
                        choices=['MMD', 'IGK'], help='Method for Domain Adaptation')
    parser.add_argument('-pt', '--prev_trained', action='store_true',
                        help='use stored model if available')
    parser.add_argument('-le','--last_epoch', action='store_true',
                        help='store model at last epoch')
    parser.add_argument('-vs', '--validate_src', action='store_true',
                        help='validation is made on source dataset not on target.')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("Hi, I'm DADER entrypoint!")
    print("Input taken from: ", args.input, args.test_data)
    print("Input directory: ", os.listdir(args.input))
    print("Output directory: ", os.listdir(args.output))

    # Step 1. Convert input data into the format expected by the method
    print("Method input: ", os.listdir(args.input))
    prefix_1 = 'tableA_'
    prefix_2 = 'tableB_'
    columns_to_join = None


    #parameters:
    #parser.add_argument('--epoch', type=int, default=0,
    #                    help="Specify the number of epochs for pretrain")

    #parser.add_argument('--epoch_path', type=str, default='new-all/',
    #  help="Specify log step size for adaptation")

    #parser.add_argument('--seed_list', type=str, default="",
    # help="Specify log step size for adaptation")
    hyperparameters = namedtuple('hyperparameters', ['model', 'num_epochs', 'batch_size',
                                                     'max_seq_length', 'alpha', 'beta', 'temperature', 'max_grad_norm',
                                                     'clip_value', 'output', 'source_only', 'log_step',
                                                     'load', 'need_pred_res', 'need_kd_model', 'adda',
                                                     'rec_lr', 'rec_epoch', 'd_learning_rate', 'model_index', 'pre_log_step',
                                                    'pre_epochs', 'last_epoch', 'validate_src'])

    hp = hyperparameters(model = args.languagemodel, num_epochs = args.epochs, batch_size = 32,
                         max_seq_length=128, alpha = 1.0, beta = 0.1, temperature = 20, max_grad_norm = 1.0,
                         clip_value = 0.01, output = args.output, source_only=False, log_step=50,
                         load=False, need_pred_res=0, need_kd_model=True, adda=0, rec_lr='', rec_epoch=False,
                         d_learning_rate=1e-6, model_index=3, pre_log_step=10, pre_epochs=5, last_epoch=args.last_epoch,
                         validate_src=args.validate_src)

    if args.adapt=='IGK':
        hp = hp._replace(log_step=10, beta=1)
        #hp.log_step=10
        #hp.beta = 1

    source_train, source_valid, target_train, target_valid, target_test = transform_input(args.input, args.test_data, columns_to_join, ' ', [prefix_1, prefix_2],
                                                   args.input_train_full, args.test_full)
    t_start = time.process_time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    source_train_features = convert_table_to_features(source_train, hp, tokenizer)
    source_valid_features = convert_table_to_features(source_valid, hp, tokenizer)
    target_train_features = convert_table_to_features(target_train, hp, tokenizer)
    target_valid_features = convert_table_to_features(target_valid, hp, tokenizer)
    source_train_data_loader = get_data_loader(source_train_features, hp.batch_size, "train")
    source_valid_data_loader = get_data_loader(source_valid_features, hp.batch_size, "dev")
    target_train_data_loader = get_data_loader(target_train_features, hp.batch_size, "train")
    target_valid_data_loader = get_data_loader(target_valid_features, hp.batch_size, "dev")
    t_p_train = time.process_time()
    target_test_features = convert_table_to_features(target_test, hp, tokenizer)
    target_test_data_loader = get_data_loader(target_test_features, hp.batch_size, "dev")
    t_p_test = time.process_time()
    t_preprocess = t_p_test + t_p_train
    f = open(os.path.join(args.output, 'prep_time.txt'), 'w')
    print(f'Prep time train: {t_p_train - t_start}, Prep time test: {t_p_test - t_p_train}', file=f)
    f.close()

    # load models
    if args.adapt == 'MMD':
        if args.languagemodel.lower() == 'bert':
            src_encoder = BertEncoder()
            src_classifier = BertClassifier()


        src_encoder = init_model(hp, src_encoder)
        src_classifier = init_model(hp, src_classifier)

        print("=== Training classifier for source domain ===")
        t_start = time.process_time()
        src_encoder, src_classifier, results_per_epoch = train_mmd(hp, src_encoder, src_classifier,
                                                                   source_train_data_loader, source_valid_data_loader,
                                                                   target_train_data_loader, target_valid_data_loader)
        t_train = time.process_time() - t_start
        t_start = time.process_time()
        f1, _, labels, predictions, l_ids, r_ids = evaluate_mmd(hp,src_encoder, src_classifier, target_test_data_loader, None, pattern=10000)
        t_test = time.process_time() - t_start
    elif args.adapt == 'IGK':
        if args.languagemodel.lower() == 'bert':
            src_encoder = BertEncoder()
            tgt_encoder = BertEncoder()
            src_classifier = BertClassifier()
        discriminator = Discriminator()

        src_encoder = init_model(hp, src_encoder)
        src_classifier = init_model(hp, src_classifier)
        tgt_encoder = init_model(hp, tgt_encoder)
        discriminator = init_model(hp, discriminator)
        only_s = 0

        # train F and M

        print("=== Training F and M ===")
        t_start = time.process_time()
        src_encoder, src_classifier, only_s, results_per_epoch_pretrain = pretrain_best(hp, src_encoder, src_classifier,
                                                                        source_train_data_loader, source_valid_data_loader,
                                                                        target_valid_data_loader, target_train_data_loader)

        for params in src_encoder.parameters():
            params.requires_grad = False
        for params in src_classifier.parameters():
            params.requires_grad = False

        best_res = -1

        print("=== Training F' and A ===")

        tgt_encoder.load_state_dict(src_encoder.state_dict())
        tgt_encoder, discriminator, best_res, best_f1, results_per_epoch_adapt = adapt_best(hp, src_encoder, tgt_encoder, discriminator,
                                                                   src_classifier, source_train_data_loader, source_valid_data_loader,
                                                                   target_train_data_loader, target_valid_data_loader)

        t_train = time.process_time() - t_start
        t_start = time.process_time()
        predictions, labels, l_ids, r_ids = evaluate_gen(tgt_encoder, src_classifier, target_test_data_loader, return_preds=True)
        print("=== Result of InvGAN+KD: ===")
        print(best_res)
        t_test = time.process_time() - t_start
        results_per_epoch = results_per_epoch_pretrain + results_per_epoch_adapt

    transform_output(predictions, labels, l_ids, r_ids, results_per_epoch, t_preprocess, t_train, t_test,
                     args.test_data, args.input,
                     source_train.shape[0], args.output)

