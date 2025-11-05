# RunPod Training Guide - LayoutLMv3 on CommonForms

Complete guide to train LayoutLMv3 on the CommonForms dataset using RunPod.

## 🚀 Quick Start (5 minutes)

### 1. Launch RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Sign up/Login
3. Click **"Deploy"** → **"GPU Instances"**
4. **Choose GPU:**
   - **Recommended:** RTX A5000 (24GB) - ~$0.34/hr
   - **Best:** A100 (40GB/80GB) - ~$1.69/hr
   - **Budget:** RTX 3090 (24GB) - ~$0.24/hr
5. **Choose Template:**
   - Use **"RunPod Pytorch"** or **"RunPod Tensorflow"** (both have CUDA)
   - Or use **"RunPod Fast Stable Diffusion"** (has most ML libs)
6. **Storage:**
   - Container Disk: 50GB
   - Volume Disk: 100GB+ (for dataset) or use streaming mode
7. Click **"Deploy On-Demand"** or **"Deploy Spot"** (cheaper but can be interrupted)

### 2. Connect to Your Pod

Once the pod is running:

**Option A: Web Terminal (easier)**

- Click **"Connect"** → **"Start Web Terminal"**

**Option B: SSH (recommended for long sessions)**

```bash
# Get SSH command from RunPod dashboard
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

**Option C: JupyterLab**

- Click **"Connect"** → **"Connect to JupyterLab"**
- Open a terminal inside JupyterLab

### 3. Run Setup Script

```bash
# Download and run the setup script
cd /workspace
wget https://raw.githubusercontent.com/andrew-luo-extend/transformers-research-projects/main/layoutlmv3/runpod_setup.sh
chmod +x runpod_setup.sh

# Set your credentials (IMPORTANT!)
export HF_TOKEN='hf_your_token_here'
export HF_USERNAME='your-hf-username'

# Run setup
bash runpod_setup.sh
```

### 4. Authenticate with HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Verify
huggingface-cli whoami
```

### 5. Start Training

```bash
cd /workspace/layoutlmv3-training/transformers-research-projects/layoutlmv3

# Quick test (5 minutes)
./train_test.sh

# If test passes, start full training in screen
screen -S training
./train_full.sh

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r training
```

---

## 📋 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Navigate to workspace
cd /workspace

# 2. Clone repository
git clone https://github.com/huggingface/transformers-research-projects.git
cd transformers-research-projects/layoutlmv3

# 3. Install dependencies
pip install -r requirements.txt
pip install transformers torch evaluate accelerate sentencepiece huggingface_hub

# 4. Login to HuggingFace
export HF_TOKEN='hf_your_token_here'
huggingface-cli login --token $HF_TOKEN

# 5. Start training
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir /workspace/output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --fp16 \
  --push_to_hub \
  --hub_model_id "your-username/layoutlmv3-commonforms" \
  --overwrite_output_dir
```

---

## 🎯 Training Options

### Option 1: Full Dataset (Recommended for Production)

```bash
./train_full.sh
```

- Downloads entire 308GB dataset
- Requires: 500GB+ volume storage
- Training time: ~24-48 hours on A100
- Best accuracy

### Option 2: Streaming Mode (Save Disk Space)

```bash
./train_streaming.sh
```

- Streams data on-the-fly (no download)
- Requires: Only ~50GB storage
- Training time: ~30-50% slower
- Same accuracy as full dataset

### Option 3: Quick Test

```bash
./train_test.sh
```

- Uses only 50 training samples
- Training time: ~5 minutes
- For testing pipeline only

---

## 📊 Monitoring Training

### Monitor GPU Usage

```bash
# In a new terminal/screen
./monitor.sh

# Or manually
watch -n 1 nvidia-smi
```

### View Training Logs

```bash
# Follow logs in real-time
tail -f /workspace/output/trainer_log.txt

# Or if using screen
screen -r training
```

### TensorBoard

```bash
# In a new screen session
screen -S tensorboard
./launch_tensorboard.sh

# Access via RunPod's TCP port forwarding
# Go to RunPod dashboard → Your Pod → "TCP Port Mappings"
# Add port 6006, then access via the provided URL
```

---

## 💾 Saving Your Work

### Automatic (Recommended)

Your model automatically uploads to HuggingFace Hub after each epoch:

- Location: `https://huggingface.co/your-username/layoutlmv3-commonforms`
- Includes: Model weights, config, training metrics

### Manual Download

```bash
# From your local machine
runpodctl receive <pod-id>:/workspace/output ./local_output

# Or use rsync
rsync -avz -e "ssh -p <port>" root@<pod-ip>:/workspace/output ./local_output
```

### To RunPod Volume (Persistent Storage)

```bash
# Copy to volume (survives pod termination)
cp -r /workspace/output /runpod-volume/
```

---

## 🔧 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python run_funsd_cord.py \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  ...
```

### Disk Space Issues

```bash
# Use streaming mode
./train_streaming.sh

# Or clear cache
rm -rf ~/.cache/huggingface/
```

### Pod Disconnected

```bash
# Reconnect via RunPod dashboard
# Then reattach to screen
screen -r training

# Training continues in background!
```

### CUDA Out of Memory

```bash
# Use gradient checkpointing (in Python script)
# Or reduce max_seq_length
python run_funsd_cord.py \
  --max_seq_length 256 \
  ...
```

---

## 💰 Cost Estimation

### RunPod Pricing (Spot/On-Demand)

| GPU       | VRAM | Spot Price | On-Demand |
| --------- | ---- | ---------- | --------- |
| RTX 3090  | 24GB | $0.24/hr   | $0.34/hr  |
| RTX A5000 | 24GB | $0.34/hr   | $0.49/hr  |
| A100      | 40GB | $1.09/hr   | $1.69/hr  |
| A100      | 80GB | $1.39/hr   | $2.09/hr  |

### Storage Pricing

- Container Disk: Free (lost on termination)
- Volume Storage: ~$0.10/GB/month
- Network: Free

### Example Cost (24 hours training)

- A100 40GB Spot: $1.09 × 24 = **$26.16**
- 500GB Volume: $50/month ÷ 30 = **$1.67/day**
- **Total: ~$28** for 24 hours

**💡 Tip:** Use Spot instances (60-70% cheaper) with screen/tmux so training continues if pod is interrupted.

---

## 🎓 Best Practices

### 1. Use Screen/Tmux

```bash
# Always run training in screen
screen -S training
./train_full.sh
# Ctrl+A, D to detach
```

### 2. Enable Auto-Upload

```bash
# Always use --push_to_hub
# Model saves automatically to HuggingFace
```

### 3. Monitor Costs

- Check RunPod dashboard regularly
- Terminate pod when done (very important!)
- Use Spot instances for cost savings

### 4. Save Checkpoints

```bash
# Script already configured to save every epoch
--save_strategy epoch
--save_total_limit 3
```

### 5. Test First

```bash
# Always run test script first
./train_test.sh
# Then run full training
```

---

## 🛑 Terminating Your Pod

**IMPORTANT:** Don't forget to terminate when done!

1. Go to RunPod dashboard
2. Find your pod
3. Click **"Terminate"**
4. Container storage is deleted (but volume persists)
5. Model is safe on HuggingFace Hub ✓

---

## 📞 Support

- RunPod Discord: https://discord.gg/runpod
- HuggingFace Forum: https://discuss.huggingface.co/
- Repository Issues: https://github.com/huggingface/transformers-research-projects/issues

---

## 📝 Configuration Options

Edit `runpod_setup.sh` to customize:

```bash
HF_TOKEN="hf_your_token"           # Your HuggingFace token
HF_USERNAME="your-username"        # Your HF username
MODEL_NAME="layoutlmv3-commonforms" # Model name on Hub
USE_STREAMING="false"              # true = streaming mode
```

Edit training scripts (`train_full.sh`, etc.) to customize hyperparameters:

- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--learning_rate`: Learning rate
- `--max_train_samples`: Limit training samples (for testing)

---

Happy Training! 🚀
