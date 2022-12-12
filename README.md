# bow-vs-graph-vs-seq-textclassification
Code for reproducing [Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP](https://arxiv.org/abs/2109.03777)

# Data
Download the ``data`` folder from the [Text-GCN](https://github.com/lgalke/text_gcn) repository and place the folder in the main directory.

# Setting up the environment
``conda create --name text_clf``

``conda activate text_clf``

``pip install -r requirements.txt``

# Glove embeddings
Download [glove.42B.300d.zip](https://nlp.stanford.edu/projects/glove/) and place it inside a folder called ``glove``.

# Scripts to run
Run experiments for all datasets by replacing the dataset name below [20ng, R8, R52, ohsumed, mr]

``DATASET="20ng"``

## tf-idf MLP
``python run_text_classification.py --model_type mlp --tokenizer_name "bert-base-uncased" --bow_aggregation "tfidf" --mlp_num_layers 1 --batch_size 16 --learning_rate "0.001" --epochs 100 --num_workers 4 --results_file "results/results_tfidf_mlp.csv" "$DATASET"``

## glove MLP, 2 layers
``python run_text_classification.py --model_type mlp --model_name_or_path "glove/glove.42B.300d.txt" --batch_size 16 --learning_rate "0.001" --mlp_num_layers 2 --mlp_embedding_dropout "0.0" --epochs 100 --num_workers 4 --results_file "results/results_glove42b_mlp.csv" "$DATASET"``

## glove MLP, 3 layers
``python run_text_classification.py --model_type mlp --model_name_or_path "glove/glove.42B.300d.txt" --batch_size 16 --learning_rate "0.001" --mlp_num_layers 3 --mlp_embedding_dropout "0.0" --epochs 100 --num_workers 4 --results_file "results/results_glove42b_mlp_2.csv" "$DATASET"``

## MLP, 1 layer
``python run_text_classification.py --model_type mlp --tokenizer_name "bert-base-uncased" --batch_size 16 --learning_rate "0.001" --epochs 100 --num_workers 4 --results_file "results/results_mlp.csv" "$DATASET"``

# MLP, 2 layers
``python run_text_classification.py --model_type mlp --tokenizer_name "bert-base-uncased" --batch_size 16 --learning_rate "0.001" --epochs 100 --mlp_num_layers 2 --num_workers 4 --results_file "results/results_mlp_2.csv" "$DATASET"``

# DistilBERT
``python run_text_classification.py --model_type distilbert --model_name_or_path "distilbert-base-uncased" --batch_size 8 --learning_rate "0.00005" --gradient_accumulation_steps 4 --epochs 10 --num_workers 4 --results_file "results/results_distilbert_10epochs.csv" "$DATASET"``

# BERT
``python run_text_classification.py --model_type bert --model_name_or_path "bert-base-uncased" --batch_size 8 --learning_rate "0.00005" --gradient_accumulation_steps 4 --epochs 10 --num_workers 4 --results_file "results/results_bert_10epochs_1.csv" "$DATASET"``

# BERT, no positional encoding
``python run_text_classification.py --model_type bert --model_name_or_path "bert-base-uncased" --batch_size 8 --learning_rate "0.00005" --gradient_accumulation_steps 4 --epochs 10 --ignore_position_ids --num_workers 4 --results_file "results/results_bert_10epochs.csv" "$DATASET"``

# BERT, with shuffle augmentation, 0.2
``python run_text_classification.py --model_type bert --model_name_or_path "bert-base-uncased" --batch_size 8 --learning_rate "0.00005" --gradient_accumulation_steps 4 --epochs 10 --shuffle_augment 0.2 --num_workers 4 --results_file "results/results_bert_10epochs.csv" "$DATASET"``
