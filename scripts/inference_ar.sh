model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=8

master_port=29500
num_process=4
batch_size=24
d_model=16
d_ff=32

comment='TimeLLM-Cryptex'


python inference_ar.py \
        --model_path ./trained_models/TimeLLM_CRYPTEX_h_512_96_MS.pth \
        --output_path ./predictions/ar_predictions.csv \
        --data_path ./dataset/cryptex/candlesticks-h.csv \
        --is_training 1 \
        --model_id CRYPTEX_h_512_96_MS \
        --task_name long_term_forecast \
        --model_comment $comment \
        --model $model_name \
        --data CRYPTEX \
        --features MS \
        --target close \
        --seq_len 512 \
        --label_len 48 \
        --pred_len 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model $d_model \
        --d_ff $d_ff \
        --factor 3 \
        --llm_layers $llama_layers \
        --prompt_domain 1 \
        --freq h
