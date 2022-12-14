# code for training and evaluation
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
import transformers
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertModel,
                          BertTokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification, DistilBertModel,
                          DistilBertTokenizer, get_linear_schedule_with_warmup)
from transformers import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,\
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer

from tokenization import build_tokenizer_for_word_embeddings
from data import load_data, load_word_vectors, shuffle_augment
from models import MultiLayerPerceptron as MLP, collate_for_mlp

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

try:
    import wandb
    WANDB = True
except ImportError:
    print("WandB not installed, to track experiments: pip install wandb")
    WANDB = False
WANDB=False

def inverse_document_frequency(encoded_docs, vocab_size):
    # returns idf scores
    # input: 
    # output: IDF scores in the shape [vocab_size]
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i,j] += 1
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)

def pad(seqs, with_token=0, to_length=None):
    # pad the sequence to the given max length
    # if max length is unspecified, pad to the length of the longest sequence in the list
    # input:
    #   seqs - list of sequences
    #   with_token - pad token
    #   to_length - max length
    # output: list of padded sequences
    if to_length is None:
        to_length = max(len(seq) for seq in seqs)
    seqs_padded = []
    for seq in seqs:
        if len(seq) >= to_length:
            seqs_padded.append(seq[:to_length])
        else:
            seqs_padded.append(seq + (to_length - len(seq)) * [with_token])
    
    return seqs_padded

def get_collate_for_transformer(pad_token_id):
    # closure to include padding in collate function
    # input: pad token id
    # output: collate function

    def _collate_for_transformer(examples):
        docs, labels = list(zip(*examples))
        input_ids = torch.tensor(pad(docs, with_token=pad_token_id))
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask[input_ids == pad_token_id] = 0
        labels = torch.tensor(labels)
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids, labels
    
    return _collate_for_transformer

def train(args, train_data, model, tokenizer, logger):
    # function to train the model given the data, tokenizer and hyperparameters
    if args.model_type == 'mlp':
        collate_fn = collate_for_mlp
    else:
        collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)

    # data loading
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=('cuda' in str(args.device)))

    # len(train_loader) no. of batches
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    writer = SummaryWriter()

    if args.ignore_position_ids:
        print("Setting position ids to zero and ignoring grad")
        if args.model_type == 'bert':
            model.bert.embeddings.position_embeddings.weight.requires_grad = False
            model.bert.embeddings.position_embeddings.weight.zero_()
        else:
            raise NotImplementedError("Ignore position ids only implemented for BERT")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size  = %d", args.batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.epochs, desc="Epoch")
    # iterating through the dataset
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type == 'mlp':
                # Batch: torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)
                outputs = model(batch[0], batch[1], batch[2])
            else:
                # Batch : input_ids, attention_mask, token_type_ids, labels
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                if args.ignore_position_ids:
                    inputs['position_ids'] = torch.zeros(
                        inputs['input_ids'].shape[0],  # bsz
                        inputs['input_ids'].shape[1],  # len
                        device=inputs['input_ids'].device,
                        dtype=torch.long
                    )
                outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if WANDB:
                    wandb.log({'epoch': epoch,
                               'lr': scheduler.get_last_lr()[0],
                               'loss': loss})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # if args.evaluate_during_training:
                #     results = evaluate(args, dev_data, model, tokenizer)
                #     for key, value in results.items():
                #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                avg_loss = (tr_loss - logging_loss)/ args.logging_steps
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', avg_loss, global_step)
                logging_loss = tr_loss

    writer.close()
    return global_step, tr_loss / global_step

def evaluate(args, dev_or_test_data, model, tokenizer):
	# evaluate the given model on the specified dataset

    if args.model_type == 'mlp':
        collate_fn = collate_for_mlp
    else:
        collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)
    data_loader = torch.utils.data.DataLoader(dev_or_test_data,
                                              collate_fn=collate_fn,
                                              num_workers=args.num_workers,
                                              batch_size=args.test_batch_size,
                                              pin_memory=('cuda' in str(args.device)),
                                              shuffle=False)

    all_logits = []
    all_targets = []
    nb_eval_steps, eval_loss = 0, 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.model_type == 'mlp':
                # batch consist of (flat_inputs, lenghts, labels)
                outputs = model(batch[0], batch[1], batch[2])
                all_targets.append(batch[2].detach().cpu())
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                all_targets.append(inputs['labels'].detach().cpu())
        nb_eval_steps += 1
        # outputs [:2] should hold loss, logits
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    eval_loss /= nb_eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size

    f1_micro = f1_score(targets, preds, average='micro')
    f1_macro = f1_score(targets, preds, average='macro', zero_division=1)

    if WANDB:
        wandb.log({"test/acc": acc, "test/loss": eval_loss,
                   "test/f1_micro": f1_micro,
                   "test/f1_macro": f1_macro})
    return acc, eval_loss


def run_xy_model(args, logger):
    # run the model
    print("Loading data...")

    if args.model_type == "mlp" and args.model_name_or_path is not None:
        print("Assuming to use word embeddings as both model_type=mlp and model_name_or_path are given")
        print("Using word embeddings -> forcing wordlevel tokenizer")
        vocab, embedding = load_word_vectors(args.model_name_or_path, unk_token="[UNK]")
        tokenizer = build_tokenizer_for_word_embeddings(vocab)
    else:
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        embedding = None
    print("Using tokenizer:", tokenizer)


    do_truncate = not (args.stats_and_exit or args.model_type == 'mlp')
    if args.stats_and_exit:
        # We only compute dataset stats including length, so NOT truncate
        max_length = None
    elif args.model_type == 'mlp':
        max_length = None
    else:
        max_length = 512 # should hold for all used transformer models?

    enc_docs, enc_labels, train_mask, test_mask, label2index = load_data(args.dataset,
                                                                         tokenizer,
                                                                         max_length=max_length,
                                                                         construct_textgraph=False,
                                                                         n_jobs=args.num_workers)


    print("Data loading done..")

    print("\nDocument statistics:")
    lens = np.array([len(doc) for doc in enc_docs])
    print("Min/max document length:", (lens.min(), lens.max()))
    print("Mean document length: {:.4f} ({:.4f})".format(lens.mean(), lens.std()))
    assert len(enc_docs) == len(enc_labels) == train_mask.size(0) == test_mask.size(0)
    enc_docs_arr, enc_labels_arr = np.array(enc_docs, dtype='object'), np.array(enc_labels)

    train_docs = enc_docs_arr[train_mask]
    train_labels = enc_labels_arr[train_mask]

    # shuffling data
    if args.shuffle_augment:
        factor = float(args.shuffle_augment)

        # Generate new permuted documents
        new_docs, new_labels = shuffle_augment(list(train_docs),
                                               list(train_labels),
                                               factor=factor,
                                               random_seed=args.seed)

        # Convert to numpy
        new_docs = np.array(new_docs, dtype='object')
        new_labels = np.array(new_labels)

        # Augment the training data
        train_docs = np.concatenate([train_docs, new_docs])
        train_labels = np.concatenate([train_labels, new_labels])

    train_data = list(zip(train_docs, train_labels))

    test_data = list(zip(enc_docs_arr[test_mask], enc_labels_arr[test_mask]))

    print("\nData statistics:")
    print("Total number of documents:", len(enc_docs))
    print("Size of train set:", len(train_data))
    print("Size of test set:", len(test_data))
    print("Number of classes:", len(label2index))

    if args.stats_and_exit:
        print("Warning: length stats depend on tokenizer and max_length of model, chose MLP to avoid trimming before computing stats.")
        exit(0)

    if args.model_type != 'mlp':
        config_class, model_class, __ = MODEL_CLASSES[args.model_type]
        print("Loading", args.model_type)
        print("Loading config")
        config = config_class.from_pretrained(args.model_name_or_path,
                                              num_labels=len(label2index),
                                              cache_dir=CACHE_DIR)

        print(config)
        print("Loading model")
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=CACHE_DIR)
        # This is a ForSequenceClassification Model
    else:
        print("Initializing MLP")

        if embedding is not None:
            # Vocab size given by embedding
            vocab_size = None
        else:
            vocab_size = tokenizer.vocab_size

        if args.bow_aggregation == 'tfidf':
            print("Using IDF")
            idf = inverse_document_frequency(enc_docs_arr[train_mask], tokenizer.vocab_size).to(args.device)
        else:
            idf = None

        model = MLP(vocab_size, len(label2index),
                    num_hidden_layers=args.mlp_num_layers,
                    hidden_size=args.mlp_hidden_size,
                    embedding_dropout=args.mlp_embedding_dropout,
                    dropout=args.mlp_dropout,
                    mode=args.bow_aggregation,
                    pretrained_embedding=embedding,
                    idf=idf,
                    bow_aggregation=args.bow_aggregation,
                    freeze=args.freeze_embedding)

    model.to(args.device)


    if WANDB:
        wandb.watch(model, log_freq=args.logging_steps)

    train(args, train_data, model, tokenizer, logger)
    acc, eval_loss = evaluate(args, test_data, model, tokenizer)
    print(f"[{args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}")
    return acc
