#!/bin/bash

# This script automatically runs inference for all trained models found
# in the specified MODELS_DIR.

MODELS_DIR="/mnt/nfs/models"

# Find all model files (.pth) and loop through them
for model_path in "$MODELS_DIR"/*.pth; do

        # Extract the filename without the directory and .pth extension
        model_filename=$(basename "$model_path")
        model_id="${model_filename%.pth}"

        python inference_ar.py --model_id "$model_id"

done