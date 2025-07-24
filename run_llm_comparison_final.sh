#!/bin/bash

# Final LLM Comparison: LLAMA, LLAMA3.1, QWEN, GEMMA with Adaptive Loss
# Purpose: Test 4 LLM models with proven comprehensive adaptive loss configuration

set +e  # Continue on errors

# Configuration
LOG_DIR="/mnt/nfs/logs/llm_comparison_final"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/llm_comparison_final_$TIMESTAMP.log"

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

log "Starting Final LLM Comparison"
log "Testing LLAMA, LLAMA3.1, QWEN, GEMMA with comprehensive adaptive loss"
log "Main log: $MAIN_LOG"

# Common parameters for all experiments (using proven working configuration)
COMMON_PARAMS="--task_name long_term_forecast --llm_layers 6 --num_tokens 1000 --seq_len 32 --pred_len 7 --patch_len 1 --features MS --granularity daily --enable_mlflow --use_enhanced --loss comprehensive --auto_confirm"

# Experiment 1: LLAMA + Comprehensive Adaptive Loss
run_experiment "exp1_llama_adaptive" \
"python launch_experiment.py --adaptive exp1_llama_adaptive --llm_model LLAMA $COMMON_PARAMS"

# Experiment 2: LLAMA3.1 + Comprehensive Adaptive Loss (may fail due to gated access)
run_experiment "exp2_llama31_adaptive" \
"python launch_experiment.py --adaptive exp2_llama31_adaptive --llm_model LLAMA3.1 $COMMON_PARAMS"

# Experiment 3: QWEN + Comprehensive Adaptive Loss
run_experiment "exp3_qwen_adaptive" \
"python launch_experiment.py --adaptive exp3_qwen_adaptive --llm_model QWEN $COMMON_PARAMS"

# Experiment 4: GEMMA + Comprehensive Adaptive Loss
run_experiment "exp4_gemma_adaptive" \
"python launch_experiment.py --adaptive exp4_gemma_adaptive --llm_model GEMMA $COMMON_PARAMS"

log "Final LLM Comparison Complete"
log "Experiment Status Summary:"
cat $LOG_DIR/experiment_status.csv | tee -a $MAIN_LOG

log "Results saved to:"
log "  Main log: $MAIN_LOG"
log "  Status CSV: $LOG_DIR/experiment_status.csv"
log "  MLFlow UI: Check for results and metrics"
log "  Models saved to: /mnt/nfs/models/"

echo "Final LLM comparison pipeline completed"