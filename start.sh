#!/bin/bash

# Set strict error handling
set -e

echo "Starting Wan Camera Serverless..."

# Check if volume is mounted
if [ ! -d "$RUNPOD_VOLUME_PATH" ]; then
    echo "ERROR: Volume not mounted at $RUNPOD_VOLUME_PATH"
    exit 1
fi

# Create directory structure in volume
mkdir -p $RUNPOD_VOLUME_PATH/models/diffusion_models
mkdir -p $RUNPOD_VOLUME_PATH/models/loras  
mkdir -p $RUNPOD_VOLUME_PATH/models/vae
mkdir -p $RUNPOD_VOLUME_PATH/models/text_encoders

# Create symlinks from ComfyUI models directory to volume
echo "Setting up model symlinks..."
cd /app/ComfyUI

# Remove existing models directory if it exists and create symlinks
rm -rf models
ln -sf $RUNPOD_VOLUME_PATH/models models

# Initialize model manager to download missing models
echo "Checking and downloading models..."
python3 /app/utils/model_manager.py

echo "Model setup complete. Starting handler..."

# Start the RunPod handler
python3 -u /app/handler.py