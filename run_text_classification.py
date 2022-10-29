# run text classification
# datasets: 20ng, R8, R52, ohsumed, mr
# models: [fill]

import csv
import itertools as it
import logging
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from tokenization import build_tokenizer_for_word_embeddings
from data import load_data, load_word_vectors, shuffle_augment
from models import MLP, collate_for_mlp
from run_model import run_xy_model

try:
    import wandb
    WANDB = True
except ImportError:
    print("WandB not installed, to track experiments: pip install wandb")
    WANDB = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()
CACHE_DIR = 'cache/textclf'
MEMORY = Memory(CACHE_DIR, verbose=2)

VALID_DATASETS = [ '20ng', 'R8', 'R52', 'ohsumed', 'mr'] + ['TREC', 'wiki']

ALL_MODELS = BERT_PRETRAINED_MODEL_ARCHIVE_LIST + DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertModel),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetModel),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMModel),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaModel),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel)
}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=VALID_DATASETS, help="")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: either mlp, bert, distilbert",
                        choices=["mlp", "distilbert", "bert"])
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Optional path to word embedding with model type 'mlp' OR huggingface shortcut name such as distilbert-base-uncased for model type 'distilbert'")
    parser.add_argument("--results_file", default=None,
                        help="Store results to this results file")

    ## Training config
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Batch size for testing (defaults to train batch size)")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument('--unfreeze_embedding', dest="freeze_embedding", default=True,
            action='store_false', help="Allow updating pretrained embeddings")

    ## Training Hyperparameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers")

    parser.add_argument("--stats_and_exit", default=False,
                        action='store_true',
                        help="Print dataset stats and exit.")

    # MLP Params
    parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of hidden layers within MLP")
    parser.add_argument("--mlp_hidden_size", default=1024, type=int, help="Hidden dimension for MLP")
    parser.add_argument("--bow_aggregation", default="mean", choices=["mean", "sum", "tfidf"],
            help="Aggregation for bag-of-words models (such as MLP)")
    parser.add_argument("--mlp_embedding_dropout", default=0.5, type=float, help="Dropout for embedding / first hidden layer ")
    parser.add_argument("--mlp_dropout", default=0.5, type=float, help="Dropout for all subsequent layers")

    parser.add_argument("--comment", help="Some comment for the experiment")
    parser.add_argument("--ignore_position_ids",
                        help="Use all zeros to pos ids",
                        default=False, action='store_true')
    parser.add_argument("--seed", default=None,
                        help="Random seed for shuffle augment")
    parser.add_argument("--shuffle_augment", type=float,
                        default=0, help="Factor for shuffle data augmentation")
    ##########################
    args = parser.parse_args()

    if args.model_type in ['mlp', 'textgcn']:
        assert args.tokenizer_name or args.model_name_or_path, "Please supply tokenizer for MLP via --tokenizer_name or provide an embedding via --model_name_or_path"
    else:
        assert args.model_name_or_path, f"Please supply --model_name_or_path for {args.model_type}"

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.test_batch_size = args.batch_size if args.test_batch_size is None else args.test_batch_size

    if WANDB:
        wandb.init(project="text-clf")
        wandb.config.update(args)

    acc = {
        'mlp': run_xy_model,
        'bert': run_xy_model,
        'distilbert': run_xy_model,
        'roberta': run_xy_model,
        'xlnet': run_xy_model
    }[args.model_type](args, logger)
    if args.results_file:
        with open(args.results_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([args.model_type,args.dataset,acc])



if __name__ == '__main__':
    main()


