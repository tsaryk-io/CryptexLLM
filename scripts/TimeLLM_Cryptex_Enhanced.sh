#!/bin/bash

# Enhanced Time-LLM training script with new features and loss functions
model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=8

master_port=29500
num_process=4
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-Cryptex-Enhanced'

# Enhanced Experiment 1: Multi-feature with asymmetric loss
echo "========================================="
echo "Enhanced Experiment 1: Multi-feature with asymmetric loss"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_ENHANCED_h_512_96_M_asymmetric \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 16 \
  --stride 8 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --loss asymmetric \
  --llm_model LLAMA

# Enhanced Experiment 2: Trading loss optimization
echo "========================================="
echo "Enhanced Experiment 2: Trading loss optimization"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_ENHANCED_d_96_24_M_trading \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features M \
  --target close \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 8 \
  --stride 4 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --loss trading_loss \
  --llm_model LLAMA

# Enhanced Experiment 3: Multi-scale features
echo "========================================="
echo "Enhanced Experiment 3: Multi-scale features"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_MULTISCALE_combined_192_48_M \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_MULTISCALE \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 12 \
  --stride 6 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size 16 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --loss robust \
  --llm_model LLAMA

# Enhanced Experiment 4: Quantile loss for uncertainty
echo "========================================="
echo "Enhanced Experiment 4: Quantile loss for uncertainty"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_ENHANCED_h_168_24_M_quantile \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 168 \
  --label_len 24 \
  --pred_len 24 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 12 \
  --stride 6 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --loss quantile \
  --llm_model LLAMA

# Enhanced Experiment 5: Different LLM comparison with enhanced features
echo "========================================="
echo "Enhanced Experiment 5: GPT2 with enhanced features"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_ENHANCED_h_384_96_M_gpt2 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 384 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 16 \
  --stride 8 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers 6 \
  --loss madl \
  --llm_model GPT2 \
  --llm_dim 768

echo "========================================="
echo "All enhanced experiments completed!"
echo "========================================="

# TODO:
# - Monitor training metrics and compare with baseline
# - Implement ensemble methods
# - Add external data sources (sentiment, macro indicators)
# - Implement regime-aware evaluation