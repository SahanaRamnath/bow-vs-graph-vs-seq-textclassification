#!/usr/bin/env bash
MODEL_TYPE="mlp"
MODEL_NAME_OR_PATH="glove/glove.6B.50d.txt"
BATCH_SIZE=16
EPOCHS=3
RESULTS_FILE="results/glove6b50d_mlp.csv"
LEARNING_RATE="0.001"

# Stop on error
set -e
for seed in 1; do
	for DATASET in "20ng"; do
		python3 run_text_classification.py --model_type "$MODEL_TYPE" --model_name_or_path "$MODEL_NAME_OR_PATH" \
			--batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --mlp_num_layers 2 --mlp_embedding_dropout "0.0" \
			--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
done
