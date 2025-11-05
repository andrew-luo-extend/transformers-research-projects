#!/bin/bash
# ============================================
# RunPod Setup Script for LayoutLMv3 Training
# ============================================

set -e  # Exit on error

echo "=========================================="
echo "RunPod LayoutLMv3 Training Setup"
echo "=========================================="

# === CONFIGURATION (EDIT THESE) ===
HF_TOKEN="${HF_TOKEN:-hf_YOUR_TOKEN_HERE}"  # Set via environment or edit here
HF_USERNAME="${HF_USERNAME:-your-username}"  # Your HuggingFace username
MODEL_NAME="layoutlmv3-commonforms"
GIT_REPO="https://github.com/andrew-luo-extend/transformers-research-projects.git"
USE_STREAMING="${USE_STREAMING:-false}"  # Set to true if using streaming mode

# === VOLUME STORAGE PATHS (RunPod Network Volume) ===
CACHE_DIR="/workspace/hf-cache"
OUTPUT_DIR="/workspace/output"
DATASETS_CACHE="$CACHE_DIR/datasets"

# === CHECK GPU ===
echo "Checking GPU availability..."
nvidia-smi || echo "Warning: nvidia-smi not found"

# === INSTALL SCREEN/TMUX ===
echo "Installing screen and tmux for persistent sessions..."
apt-get update -qq && apt-get install -y -qq screen tmux > /dev/null 2>&1 || echo "Note: screen/tmux may already be installed"

# === SETUP WORKSPACE ===
echo "Setting up workspace..."
cd /workspace || cd ~
WORK_DIR="/workspace/layoutlmv3-training"
mkdir -p $WORK_DIR
cd $WORK_DIR

# === CLONE REPOSITORY ===
if [ ! -d "transformers-research-projects" ]; then
    echo "Cloning repository..."
    git clone $GIT_REPO
else
    echo "Repository already exists, pulling latest changes..."
    cd transformers-research-projects
    git pull
    cd ..
fi

cd transformers-research-projects/layoutlmv3

# === PYTHON ENVIRONMENT ===
echo "Setting up Python environment..."

# Check if we should use system Python or venv
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python not found!"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Install/upgrade pip
$PYTHON_CMD -m pip install --upgrade pip

# === INSTALL DEPENDENCIES ===
echo "Installing dependencies..."
pip install -r requirements.txt
pip install transformers torch evaluate accelerate sentencepiece huggingface_hub tensorboard

# Verify PyTorch CUDA
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# === SETUP VOLUME STORAGE ===
echo "Setting up volume storage directories..."
mkdir -p $CACHE_DIR
mkdir -p $DATASETS_CACHE
mkdir -p $OUTPUT_DIR

# Set environment variables to use volume storage
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$DATASETS_CACHE
export TRANSFORMERS_CACHE=$CACHE_DIR

# Add to bashrc for persistence
echo "export HF_HOME=$CACHE_DIR" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=$DATASETS_CACHE" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$CACHE_DIR" >> ~/.bashrc

echo "✓ Cache directories created on volume:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"

# Clear any existing cache on container disk to avoid conflicts
if [ -d "/root/.cache/huggingface" ]; then
    echo "Clearing container disk cache to prevent disk quota issues..."
    rm -rf /root/.cache/huggingface/*
fi

# === HUGGINGFACE AUTHENTICATION ===
echo "Setting up HuggingFace authentication..."
if [ "$HF_TOKEN" != "hf_YOUR_TOKEN_HERE" ]; then
    export HF_TOKEN=$HF_TOKEN
    huggingface-cli login --token $HF_TOKEN
    echo "✓ HuggingFace login successful"
else
    echo "⚠ WARNING: HF_TOKEN not set. Please set it manually:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo "  huggingface-cli login --token \$HF_TOKEN"
fi

# === CREATE TRAINING SCRIPT ===
echo "Creating training script..."
cat > train_full.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Use volume storage (CRITICAL - avoids container disk quota errors)
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf-cache

# Load HF token if available
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN=$HF_TOKEN
fi

cd /workspace/layoutlmv3-training/transformers-research-projects/layoutlmv3

echo "Starting full training..."
echo "Cache directory: /workspace/hf-cache (volume storage)"
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --logging_steps 100 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --fp16 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --push_to_hub \
  --hub_model_id "$HF_USERNAME/$MODEL_NAME" \
  --hub_strategy every_save \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --overwrite_output_dir

echo "Training completed!"
echo "Model saved to: /workspace/output"
SCRIPT_EOF

chmod +x train_full.sh

# === CREATE QUICK TEST SCRIPT ===
cat > train_test.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Use volume storage (CRITICAL - avoids container disk quota errors)
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf-cache

cd /workspace/layoutlmv3-training/transformers-research-projects/layoutlmv3

echo "Starting quick test run..."
echo "Cache directory: /workspace/hf-cache (volume storage)"
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/output_test \
  --do_train \
  --do_eval \
  --max_train_samples 50 \
  --max_eval_samples 20 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --logging_steps 5 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --fp16 \
  --overwrite_output_dir

echo "Test completed!"
SCRIPT_EOF

chmod +x train_test.sh

# === CREATE STREAMING MODE SCRIPT ===
cat > train_streaming.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Use volume storage for model outputs
export HF_HOME=/workspace/hf-cache
export TRANSFORMERS_CACHE=/workspace/hf-cache

# Load HF token if available
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN=$HF_TOKEN
fi

cd /workspace/layoutlmv3-training/transformers-research-projects/layoutlmv3

echo "Starting training with streaming mode (no dataset download)..."
echo "Output directory: /workspace/output (volume storage)"
python run_funsd_cord.py \
  --dataset_name commonforms \
  --use_streaming \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --logging_steps 100 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --fp16 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --push_to_hub \
  --hub_model_id "$HF_USERNAME/$MODEL_NAME" \
  --hub_strategy every_save \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --overwrite_output_dir

echo "Training completed!"
echo "Model saved to: /workspace/output"
SCRIPT_EOF

chmod +x train_streaming.sh

# === CREATE MONITORING SCRIPT ===
cat > monitor.sh << 'MONITOR_EOF'
#!/bin/bash

echo "GPU Monitoring (Press Ctrl+C to stop)"
watch -n 1 nvidia-smi
MONITOR_EOF

chmod +x monitor.sh

# === CREATE TENSORBOARD LAUNCH SCRIPT ===
cat > launch_tensorboard.sh << 'TB_EOF'
#!/bin/bash

echo "Starting TensorBoard..."
echo "Access at: http://localhost:6006"
echo "Or use RunPod's port forwarding"

tensorboard --logdir /workspace/output --host 0.0.0.0 --port 6006
TB_EOF

chmod +x launch_tensorboard.sh

# === DISPLAY INSTRUCTIONS ===
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Set your HuggingFace credentials:"
echo "  export HF_TOKEN='hf_your_token_here'"
echo "  export HF_USERNAME='your-hf-username'"
echo "  huggingface-cli login --token \$HF_TOKEN"
echo ""
echo "Available scripts:"
echo "  ./train_test.sh          - Quick test with 50 samples (~5 min)"
echo "  ./train_full.sh          - Full training (downloads dataset)"
echo "  ./train_streaming.sh     - Streaming mode (saves disk space)"
echo "  ./monitor.sh             - Monitor GPU usage"
echo "  ./launch_tensorboard.sh  - Launch TensorBoard"
echo ""
echo "Recommended workflow:"
echo "  1. Run test: ./train_test.sh"
echo "  2. If successful, start persistent session:"
echo "     tmux new -s training  (or: screen -S training)"
echo "  3. Inside tmux: ./train_full.sh"
echo "  4. Detach: Ctrl+B then D  (screen: Ctrl+A then D)"
echo "  5. Reattach: tmux attach -t training  (screen: screen -r training)"
echo ""
echo "Storage locations (ALL on network volume - survives pod restart):"
echo "  Dataset cache: /workspace/hf-cache/datasets"
echo "  Model output:  /workspace/output"
echo "  HuggingFace:   https://huggingface.co/$HF_USERNAME/$MODEL_NAME"
echo ""
echo "✓ All scripts configured to use volume storage (no disk quota errors)"
echo ""
echo "=========================================="

