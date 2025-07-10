#!/bin/bash

# Multi-Scale Time-LLM training script with hierarchical forecasting and ensemble methods
model_name=MultiScaleTimeLLM
train_epochs=15
learning_rate=0.005
llama_layers=8

master_port=29500
num_process=4
batch_size=16  # Smaller batch for complex architecture
d_model=64     # Adjusted for multi-scale complexity
d_ff=256

comment='MultiScale-TimeLLM-Hierarchical-Ensemble'

echo "========================================="
echo "MULTI-SCALE TIME-LLM TRAINING PIPELINE"
echo "========================================="

# Create necessary directories
mkdir -p ./trained_models/multiscale
mkdir -p ./results/multiscale
mkdir -p ./logs/multiscale

# Experiment 1: Multi-timeframe hierarchical forecasting
echo "========================================="
echo "Experiment 1: Multi-timeframe hierarchical forecasting"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id MULTISCALE_hierarchical_1h_4h_1D_192_96 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_MULTISCALE \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 60 \
  --dec_in 60 \
  --c_out 60 \
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
  --loss robust \
  --llm_model LLAMA \
  --timeframes 1h,4h,1D \
  --pred_horizons 24,96,168 \
  --multi_patch_lengths 4,8,16

# Experiment 2: Temporal attention with ensemble LLMs
echo "========================================="
echo "Experiment 2: Temporal attention with ensemble LLMs"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id MULTISCALE_ensemble_attention_1h_96_24 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 96 \
  --label_len 24 \
  --pred_len 24 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 6 \
  --stride 3 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --loss trading_loss \
  --llm_model LLAMA \
  --enable_temporal_attention 1 \
  --attention_heads 8

# Experiment 3: Adaptive multi-resolution patches
echo "========================================="
echo "Experiment 3: Adaptive multi-resolution patches"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id MULTISCALE_adaptive_patches_D_168_48 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features M \
  --target close \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 48 \
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
  --loss asymmetric \
  --llm_model LLAMA \
  --multi_patch_lengths 6,12,24 \
  --adaptive_patching 1

# Experiment 4: Cross-timeframe ensemble (multiple LLMs)
echo "========================================="
echo "Experiment 4: Cross-timeframe ensemble (LLAMA + GPT2 + BERT)"
echo "========================================="

# Train LLAMA component
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENSEMBLE_llama_h_384_96 \
  --model_comment "${comment}_LLAMA" \
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
  --batch_size 12 \
  --learning_rate 0.003 \
  --llm_layers $llama_layers \
  --loss quantile \
  --llm_model LLAMA

# Train GPT2 component
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENSEMBLE_gpt2_h_384_96 \
  --model_comment "${comment}_GPT2" \
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
  --batch_size 24 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --loss quantile \
  --llm_model GPT2 \
  --llm_dim 768

# Train BERT component  
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENSEMBLE_bert_h_384_96 \
  --model_comment "${comment}_BERT" \
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
  --batch_size 24 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --loss quantile \
  --llm_model BERT \
  --llm_dim 768

# Experiment 5: Full multi-scale system (all components)
echo "========================================="
echo "Experiment 5: Full multi-scale system integration"
echo "========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id MULTISCALE_FULL_integrated_system \
  --model_comment "${comment}_FULL" \
  --model $model_name \
  --data CRYPTEX_MULTISCALE \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 256 \
  --label_len 64 \
  --pred_len 96 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --d_model 96 \
  --d_ff 384 \
  --factor 3 \
  --patch_len 16 \
  --stride 8 \
  --itr 1 \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 0.002 \
  --llm_layers $llama_layers \
  --loss robust \
  --llm_model LLAMA \
  --timeframes 15min,1h,4h,1D \
  --pred_horizons 6,24,96,168 \
  --multi_patch_lengths 4,8,16,32 \
  --enable_temporal_attention 1 \
  --enable_adaptive_fusion 1 \
  --enable_hierarchical_prediction 1 \
  --attention_heads 12

echo "========================================="
echo "Running ensemble combination and evaluation"
echo "========================================="

# Run ensemble evaluation script (if available)
if [ -f "evaluate_ensemble.py" ]; then
    python evaluate_ensemble.py \
        --model_paths ./trained_models/multiscale/ \
        --test_data ./dataset/cryptex/candlesticks-h.csv \
        --output_dir ./results/multiscale/ \
        --ensemble_methods weighted,voting,stacking
fi

echo "========================================="
echo "Multi-scale training pipeline completed!"
echo "========================================="

echo "Trained models:"
echo "1. Hierarchical multi-timeframe forecasting"
echo "2. Temporal attention with ensemble LLMs"  
echo "3. Adaptive multi-resolution patches"
echo "4. Cross-timeframe ensemble (LLAMA+GPT2+BERT)"
echo "5. Full integrated multi-scale system"

echo ""
echo "Key innovations implemented:"
echo "• Hierarchical forecasting across 6, 24, 96, 168 step horizons"
echo "• Temporal attention focusing on relevant historical periods"
echo "• Multi-LLM ensemble with dynamic weighting"
echo "• Cross-timeframe fusion with adaptive importance"
echo "• Multi-resolution patch embeddings with adaptive selection"
echo "• Prediction reconciliation across temporal hierarchy"

echo ""
echo "Results saved to:"
echo "• Models: ./trained_models/multiscale/"
echo "• Logs: ./logs/multiscale/"
echo "• Results: ./results/multiscale/"

echo ""
echo "Next steps:"
echo "1. Evaluate ensemble performance vs individual models"
echo "2. Analyze attention maps for interpretability"
echo "3. Test uncertainty quantification capabilities"
echo "4. Compare with baseline Time-LLM performance"