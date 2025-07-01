#!/bin/bash

LLM_MODEL="LLAMA3.1"
LLM_LAYERS=10
GRANULARITY="daily"
LOSS="MSE"
METRIC="MAE"
TASK_NAME="short_term_forecast"
FEATURE_SETS=("MS" "S")
NUM_TOKENS_LIST=(100 500 1000)
SEQ_LEN_LIST=(168 91 14)
PRED_LEN_LIST=(1 7)

for feature in "${FEATURE_SETS[@]}"; do
  for num_tokens in "${NUM_TOKENS_LIST[@]}"; do
    for seq_len in "${SEQ_LEN_LIST[@]}"; do
      for pred_len in "${PRED_LEN_LIST[@]}"; do

        if [ "$seq_len" -eq 168 ]; then
          patch_len=6
          stride=4
        else
          patch_len=1
          stride=1
        fi

        python launch_experiment.py \
          --llm_model "$LLM_MODEL" \
          --llm_layers "$LLM_LAYERS" \
          --granularity "$GRANULARITY" \
          --loss "$LOSS" \
          --metric "$METRIC" \
          --task_name "$TASK_NAME" \
          --features "$feature" \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --patch_len "$patch_len" \
          --stride "$stride" \
          --num_tokens "$num_tokens" \
          --auto_confirm

      done
    done
  done
done
