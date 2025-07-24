#!/bin/bash

# Day 1: LLM Architecture Comparison with Adaptive Loss
# Purpose: Test 5 LLM models to determine best performing architecture

set +e  # Continue on errors

# Configuration
LOG_DIR="/mnt/nfs/logs/day1_llm_comparison"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/day1_llm_comparison_$TIMESTAMP.log"

# Create log directory
mkdir -p $LOG_DIR

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $MAIN_LOG
}

# Function to run experiment with error handling and skip on failure
run_experiment() {
    local exp_name=$1
    local cmd=$2
    
    log "Starting Experiment: $exp_name"
    log "Command: $cmd"
    
    start_time=$(date +%s)
    
    if eval $cmd; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log "$exp_name completed successfully in ${duration}s"
        echo "$exp_name,SUCCESS,$duration" >> $LOG_DIR/experiment_status.csv
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log "$exp_name FAILED after ${duration}s - SKIPPING AND CONTINUING"
        echo "$exp_name,FAILED,$duration" >> $LOG_DIR/experiment_status.csv
    fi
    
    log "Finished Experiment: $exp_name"
    echo "" | tee -a $MAIN_LOG
}

# Initialize status CSV
echo "experiment,status,duration_seconds" > $LOG_DIR/experiment_status.csv

log "Starting Day 1: LLM Architecture Comparison"
log "Testing 5 LLM models with adaptive loss"
log "Main log: $MAIN_LOG"

# Common parameters for all experiments
COMMON_PARAMS="--task_name long_term_forecast --llm_layers 6 --num_tokens 1000 --seq_len 32 --pred_len 7 --patch_len 1 --features MS --granularity daily --enable_mlflow --use_enhanced --loss comprehensive --auto_confirm"

# Experiment 1: QWEN + Comprehensive Adaptive Loss
run_experiment "exp1_qwen_adaptive" \
"python launch_experiment.py --adaptive exp1_qwen_adaptive --llm_model QWEN $COMMON_PARAMS"

# Experiment 2: MISTRAL + Comprehensive Adaptive Loss
run_experiment "exp2_mistral_adaptive" \
"python launch_experiment.py --adaptive exp2_mistral_adaptive --llm_model MISTRAL $COMMON_PARAMS"

# Experiment 3: GEMMA + Comprehensive Adaptive Loss
run_experiment "exp3_gemma_adaptive" \
"python launch_experiment.py --adaptive exp3_gemma_adaptive --llm_model GEMMA $COMMON_PARAMS"

# Experiment 4: GPT2 + Comprehensive Adaptive Loss (as backup)
run_experiment "exp4_gpt2_adaptive" \
"python launch_experiment.py --adaptive exp4_gpt2_adaptive --llm_model GPT2 $COMMON_PARAMS"

log "Day 1 LLM Architecture Comparison Complete"
log "Experiment Status Summary:"
cat $LOG_DIR/experiment_status.csv | tee -a $MAIN_LOG

log "Results saved to:"
log "  Main log: $MAIN_LOG"
log "  Status CSV: $LOG_DIR/experiment_status.csv"
log "  MLFlow UI: Check for results and metrics"

echo "Day 1 LLM comparison pipeline completed"