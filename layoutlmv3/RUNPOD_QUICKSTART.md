# RunPod Quick Start - LayoutLMv3 CommonForms Training

## 🚀 Copy-Paste Commands for RunPod

### Step 1: Setup (Run Once)

```bash
cd /workspace

# Clone repository
git clone https://github.com/andrew-luo-extend/transformers-research-projects.git
cd transformers-research-projects/layoutlmv3

# Install dependencies
pip install -r requirements.txt
pip install transformers torch evaluate accelerate sentencepiece huggingface_hub tensorboard

# Install tmux for persistent sessions
apt-get update && apt-get install -y tmux

# Set up volume storage (CRITICAL - avoids disk quota errors)
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf-cache

# Make permanent
echo 'export HF_HOME=/workspace/hf-cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/hf-cache/datasets' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/hf-cache' >> ~/.bashrc

# Create directories
mkdir -p /workspace/hf-cache/datasets
mkdir -p /workspace/output
```

### Step 2: HuggingFace Login

```bash
# Set your credentials
export HF_TOKEN='hf_xxxxxxxxxxxxx'  # YOUR TOKEN HERE
export HF_USERNAME='your-username'   # YOUR USERNAME HERE

# Login
huggingface-cli login --token $HF_TOKEN

# Verify
huggingface-cli whoami
```

### Step 3: Quick Test (5 minutes)

```bash
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/test_output \
  --do_train --do_eval \
  --max_train_samples 20 \
  --max_eval_samples 10 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --logging_steps 2 \
  --fp16 \
  --overwrite_output_dir
```

### Step 4: Full Training (if test passes)

```bash
# Start persistent session
tmux new -s training

# Full training with dataset download to volume
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/output \
  --do_train --do_eval \
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
  --hub_model_id "$HF_USERNAME/layoutlmv3-commonforms" \
  --hub_strategy every_save \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --overwrite_output_dir

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## 🎯 Alternative: Streaming Mode (No Download)

Saves disk space, ~30% slower:

```bash
tmux new -s training

python run_funsd_cord.py \
  --dataset_name commonforms \
  --use_streaming \
  --model_name_or_path microsoft/layoutlmv3-base \
  --cache_dir /workspace/hf-cache \
  --output_dir /workspace/output \
  --do_train --do_eval \
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
  --hub_model_id "$HF_USERNAME/layoutlmv3-commonforms" \
  --hub_strategy every_save \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --overwrite_output_dir
```

---

## 📊 Monitoring

### GPU Usage

```bash
# In new terminal
watch -n 1 nvidia-smi
```

### Training Progress

```bash
# Reattach to tmux
tmux attach -t training

# Or view logs
tail -f /workspace/output/trainer_log.txt
```

### Disk Usage

```bash
# Check volume usage
df -h /workspace

# Check cache size
du -sh /workspace/hf-cache
```

---

## 🔧 Tmux Quick Reference

| Command                         | Action                               |
| ------------------------------- | ------------------------------------ |
| `tmux new -s training`          | Create new session named "training"  |
| `Ctrl+B, then D`                | Detach from session (keeps running)  |
| `tmux attach -t training`       | Reattach to session                  |
| `tmux ls`                       | List all sessions                    |
| `tmux kill-session -t training` | Kill session                         |
| `exit`                          | Exit and close session (inside tmux) |

---

## ⚠️ Important Notes

### Volume Storage

✅ **Everything saves to `/workspace/` (network volume)**

- Dataset cache: `/workspace/hf-cache/datasets` (~308GB)
- Model checkpoints: `/workspace/output`
- Survives pod termination

❌ **Avoid `/root/` (container disk - limited to ~50GB)**

### Key Arguments

- `--cache_dir /workspace/hf-cache` - Use volume (REQUIRED)
- `--eval_strategy epoch` - NOT `--evaluation_strategy`
- `--use_streaming` - No download (slower training)
- `--push_to_hub` - Auto-upload to HuggingFace

### Troubleshooting

**Disk quota exceeded:**

```bash
# Check environment variables
echo $HF_DATASETS_CACHE
# Should be: /workspace/hf-cache/datasets

# If wrong, set them:
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export HF_HOME=/workspace/hf-cache
```

**Out of memory:**

```bash
# Reduce batch size
--per_device_train_batch_size 4
--gradient_accumulation_steps 4
```

---

## 📍 File Locations

```
/workspace/
├── hf-cache/
│   ├── datasets/          # 308GB dataset cache
│   └── models/            # Model cache
├── output/                # Training checkpoints
│   ├── checkpoint-XXX/
│   └── trainer_log.txt
└── transformers-research-projects/
    └── layoutlmv3/
        └── run_funsd_cord.py
```

---

## 🎉 After Training Completes

Your model is automatically uploaded to:

```
https://huggingface.co/your-username/layoutlmv3-commonforms
```

To use it:

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "your-username/layoutlmv3-commonforms"
)
```

---

## 💰 Cost Optimization

- Use **Spot instances** (60-70% cheaper)
- Use **streaming mode** if disk is limited
- **Terminate pod** when done (very important!)
- Volume persists (can restart pod and resume)

---

**Need help?** See full guide: `RUNPOD_GUIDE.md`
