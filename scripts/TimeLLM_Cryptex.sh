model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=8

master_port=29500
num_process=4
batch_size=24
d_model=32 # 16
d_ff=128 # 32

comment='TimeLLM-Cryptex'

# Things to change:
# 1. seq_len, pred_len
# 2. model_id
# 3. task_name
# 4. features

# Experiment 1 (gpu1 - 16Gb): long_term_forecast, 512 seq_len, 96 pred_len
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_h_512_96_MS \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features MS \
  --target close \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# Experiment 2 (gpu2 - 32Gb): long_term_forecast, 512 seq_len, 96 pred_len
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_Min_512_96_M \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-Min.csv \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size 64 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# Experiment 3 (gpu4 - 16Gb): short_term_forecast, 24 seq_len, 6 pred_len, patch_len 1, stride 1
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_d_24_6_M \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 6 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 1 \
  --stride 1 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# Experiment 4 (gpu4 - 16Gb): short_term_forecast, 24 seq_len, 6 pred_len, patch_len 1, stride 1
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_d_24_6_MS \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 6 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 1 \
  --stride 1 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# Experiment 5 (gpu4 - 16Gb): short_term_forecast, 96 seq_len, 24 pred_len, patch_len 1, stride 1
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_d_96_24_M \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-D.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --patch_len 1 \
  --stride 1 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# Experiment 6 (gpu4 - 16Gb): long_term_forecast, 96 seq_len, 192 pred_len
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CRYPTEX_h_96_192_M \
  --model_comment $comment \
  --model $model_name \
  --data CRYPTEX \
  --root_path ./dataset/cryptex/ \
  --data_path candlesticks-h.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --factor 3 \
  --itr 1 \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \


# TO DO:
# - Report performances of short-term-forecasting using SMAPE, MASE and OWA
# - Implement sentiment analysis
# - Implement inference script
# - Actually cite the bolded paper citations