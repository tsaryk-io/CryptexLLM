#!/bin/bash

# Enhanced Domain-Specific Prompting Training Script for TimeLLM
model_name=TimeLLM
train_epochs=15
learning_rate=0.005
llama_layers=8

master_port=29500
num_process=4
batch_size=24
d_model=64
d_ff=256

comment='Enhanced-Domain-Prompting-TimeLLM'

echo "==========================================="
echo "DOMAIN-SPECIFIC PROMPTING TRAINING PIPELINE"
echo "==========================================="

# Create necessary directories
mkdir -p ./trained_models/enhanced_prompting
mkdir -p ./results/enhanced_prompting
mkdir -p ./logs/enhanced_prompting

# Experiment 1: Basic Enhanced Prompting (CRYPTEX_ENHANCED dataset)
echo "==========================================="
echo "Experiment 1: Enhanced Dataset with Advanced Prompting"
echo "==========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_basic_h_96_24 \
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
  --enc_in 68 \
  --dec_in 68 \
  --c_out 68 \
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
  --llm_model LLAMA \
  --prompt_domain 1

# Experiment 2: External Data with Context-Aware Prompting
echo "==========================================="
echo "Experiment 2: External Data with Context-Aware Prompting"
echo "==========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_external_h_168_48 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_EXTERNAL \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 100 \
  --dec_in 100 \
  --c_out 100 \
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
  --loss asymmetric \
  --llm_model LLAMA \
  --prompt_domain 1

# Experiment 3: Regime-Aware Prompting
echo "==========================================="
echo "Experiment 3: Regime-Aware Dynamic Prompting"
echo "==========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_regime_aware_h_192_96 \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX_REGIME_AWARE \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 85 \
  --dec_in 85 \
  --c_out 85 \
  --d_model 96 \
  --d_ff 384 \
  --factor 3 \
  --patch_len 16 \
  --stride 8 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size 12 \
  --learning_rate 0.003 \
  --llm_layers $llama_layers \
  --loss robust \
  --llm_model LLAMA \
  --prompt_domain 1

# Experiment 4: Multi-Scale with Enhanced Prompting
echo "==========================================="
echo "Experiment 4: Multi-Scale Architecture with Enhanced Prompting"
echo "==========================================="

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_multiscale_h_256_96 \
  --model_comment $comment \
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
  --train_epochs $train_epochs \
  --batch_size 8 \
  --learning_rate 0.002 \
  --llm_layers $llama_layers \
  --loss quantile \
  --llm_model LLAMA \
  --prompt_domain 1

# Experiment 5: Trading Strategy-Specific Prompting Tests
echo "==========================================="
echo "Experiment 5: Trading Strategy-Specific Prompting"
echo "==========================================="

# Scalping-focused model
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_scalping_h_24_6 \
  --model_comment "${comment}_scalping" \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 6 \
  --enc_in 68 \
  --dec_in 68 \
  --c_out 68 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 4 \
  --stride 2 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size 32 \
  --learning_rate 0.008 \
  --llm_layers $llama_layers \
  --loss trading_loss \
  --llm_model LLAMA \
  --prompt_domain 1

# Swing trading-focused model
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_swing_h_168_48 \
  --model_comment "${comment}_swing" \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --target close \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 68 \
  --dec_in 68 \
  --c_out 68 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 12 \
  --stride 6 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size 16 \
  --learning_rate 0.004 \
  --llm_layers $llama_layers \
  --loss sharpe \
  --llm_model LLAMA \
  --prompt_domain 1

# Position trading-focused model
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ENHANCED_PROMPTING_position_D_168_48 \
  --model_comment "${comment}_position" \
  --model $model_name \
  --data CRYPTEX_ENHANCED \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features M \
  --target close \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 68 \
  --dec_in 68 \
  --c_out 68 \
  --d_model 96 \
  --d_ff 384 \
  --factor 3 \
  --patch_len 24 \
  --stride 12 \
  --itr 1 \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --loss robust \
  --llm_model LLAMA \
  --prompt_domain 1

echo "==========================================="
echo "Running prompt performance evaluation"
echo "==========================================="

# Run prompt evaluation script (if available)
if [ -f "evaluate_prompting.py" ]; then
    python evaluate_prompting.py \
        --model_paths ./trained_models/enhanced_prompting/ \
        --test_data ./dataset/cryptex/candlesticks-h.csv \
        --output_dir ./results/enhanced_prompting/ \
        --evaluation_metrics mse,mae,mape,trading_return
fi

echo "==========================================="
echo "Enhanced prompting training pipeline completed!"
echo "==========================================="

echo "Trained models with enhanced prompting:"
echo "1. Basic enhanced dataset with advanced technical indicators"
echo "2. External data integration with multi-source context"
echo "3. Regime-aware dynamic prompting with market classification"
echo "4. Multi-scale architecture with hierarchical prompting"
echo "5. Trading strategy-specific models (scalping, swing, position)"

echo ""
echo "Key prompting innovations implemented:"
echo "• Market regime-aware prompt generation (9 regime types)"
echo "• Trading strategy-specific instruction templates"
echo "• External data context integration (sentiment, macro, on-chain)"
echo "• Dynamic prompt adaptation based on market conditions"
echo "• Temporal and session-aware prompting"
echo "• Risk-adjusted prediction guidance"
echo "• Multi-timeframe confluence analysis"
echo "• Performance-based prompt optimization"

echo ""
echo "Results saved to:"
echo "• Models: ./trained_models/enhanced_prompting/"
echo "• Logs: ./logs/enhanced_prompting/"
echo "• Results: ./results/enhanced_prompting/"

echo ""
echo "Next steps:"
echo "1. Compare prompt-enhanced vs baseline model performance"
echo "2. Analyze regime-specific prediction accuracy"
echo "3. Test trading strategy-specific model effectiveness"
echo "4. Optimize prompt templates based on performance results"
echo "5. Implement real-time prompt adaptation for live trading"