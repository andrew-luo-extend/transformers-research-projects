# LayoutLMv3 CommonForms Training

Fine-tuning LayoutLMv3 for object detection on the CommonForms dataset.

## What's New

This modified version of `run_funsd_cord.py` adds support for:

✅ **CommonForms Dataset** - Object detection format (bboxes, category_id)
✅ **Streaming Mode** - Handle 308GB dataset without downloading
✅ **Automatic Format Conversion** - Object detection → Token classification
✅ **HuggingFace Hub Integration** - Auto-upload models during training

## Modifications Made

### 1. Dataset Support
Added `commonforms` as a dataset option alongside `funsd` and `cord`:
- Loads from `jbarrow/CommonForms` on HuggingFace Hub
- Automatically converts object detection format to token classification

### 2. Format Conversion
Objects format:
```python
{
  "id": [0, 1],
  "bbox": [[x, y, w, h], ...],
  "category_id": [0, 0],
  ...
}
```

Converted to:
```python
{
  "words": ["[OBJ]", "[OBJ]"],
  "bboxes": [[x0, y0, x1, y1], ...],
  "ner_tags": [0, 0]
}
```

### 3. Streaming Mode
New flag `--use_streaming` to handle large datasets:
- Streams data on-the-fly
- Avoids downloading 308GB dataset
- Trades speed for disk space

### 4. Dependencies Updated
- Changed `load_metric` → `evaluate.load()` (for newer datasets library)
- Added `evaluate` package requirement

## Quick Start

### Local Testing
```bash
cd layoutlmv3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install transformers torch evaluate accelerate sentencepiece

# Quick test
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir ./test_output \
  --do_eval \
  --max_eval_samples 2 \
  --per_device_eval_batch_size 1 \
  --overwrite_output_dir
```

### Cloud Training (RunPod/Lambda Labs)

**RunPod (Recommended):**
```bash
cd /workspace
wget https://raw.githubusercontent.com/<your-repo>/main/layoutlmv3/runpod_setup.sh
export HF_TOKEN='hf_your_token'
export HF_USERNAME='your-username'
bash runpod_setup.sh
```

See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) for complete instructions.

**Lambda Labs:**
```bash
# SSH into instance
ssh ubuntu@<instance-ip>

# Setup
git clone <your-repo>
cd transformers-research-projects/layoutlmv3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install transformers torch evaluate accelerate sentencepiece
huggingface-cli login

# Train
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir ./output \
  --do_train --do_eval \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --fp16 \
  --push_to_hub \
  --hub_model_id "your-username/layoutlmv3-commonforms"
```

## Usage

### Available Datasets
```bash
# FUNSD (form understanding)
--dataset_name funsd

# CORD (receipts)
--dataset_name cord

# CommonForms (our dataset - object detection)
--dataset_name commonforms
```

### Training Modes

**Full Dataset (requires 500GB storage):**
```bash
python run_funsd_cord.py \
  --dataset_name commonforms \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir ./output \
  --do_train --do_eval \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id "username/model-name"
```

**Streaming Mode (saves disk space):**
```bash
python run_funsd_cord.py \
  --dataset_name commonforms \
  --use_streaming \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir ./output \
  --do_train --do_eval \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id "username/model-name"
```

**Quick Test:**
```bash
python run_funsd_cord.py \
  --dataset_name commonforms \
  --max_train_samples 100 \
  --max_eval_samples 20 \
  --do_train --do_eval \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4
```

## Key Arguments

### Dataset
- `--dataset_name`: funsd, cord, or commonforms
- `--use_streaming`: Enable streaming mode for large datasets
- `--max_train_samples`: Limit training samples (for testing)
- `--max_eval_samples`: Limit eval samples

### Model
- `--model_name_or_path`: Base model (default: microsoft/layoutlmv3-base)
- `--output_dir`: Where to save checkpoints

### Training
- `--num_train_epochs`: Number of epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Accumulate gradients
- `--learning_rate`: Learning rate (default: 5e-5)
- `--fp16`: Enable mixed precision training
- `--save_strategy`: When to save (steps/epoch/no)
- `--evaluation_strategy`: When to evaluate

### HuggingFace Hub
- `--push_to_hub`: Auto-upload to HuggingFace
- `--hub_model_id`: Your model name on Hub
- `--hub_strategy`: Upload strategy (every_save/end)
- `--hub_private_repo`: Public or private

## File Structure

```
layoutlmv3/
├── run_funsd_cord.py          # Modified training script
├── requirements.txt            # Python dependencies
├── runpod_setup.sh            # RunPod setup script
├── RUNPOD_GUIDE.md            # Complete RunPod guide
├── README_COMMONFORMS.md      # This file
└── venv/                      # Virtual environment (local only)
```

## Technical Details

### Data Flow

1. **Load Dataset**: `load_dataset("jbarrow/CommonForms")`
2. **Convert Format**: `convert_commonforms_to_token_format()`
   - Extract bboxes from objects dict
   - Convert [x,y,w,h] → [x0,y0,x1,y1]
   - Use category_id as labels
   - Create placeholder "[OBJ]" tokens
3. **Process**: LayoutLMv3Processor handles tokenization
4. **Train**: Standard HuggingFace Trainer
5. **Upload**: Auto-upload to Hub if --push_to_hub

### Memory Requirements

| Batch Size | Seq Length | GPU Memory | Recommended GPU |
|------------|------------|------------|-----------------|
| 4 | 512 | ~16GB | RTX 3090 |
| 8 | 512 | ~24GB | A5000, RTX 4090 |
| 16 | 512 | ~40GB | A100 40GB |
| 32 | 512 | ~80GB | A100 80GB |

Use `--gradient_accumulation_steps` to simulate larger batch sizes.

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 4
# Or use gradient accumulation
--gradient_accumulation_steps 4
# Or reduce sequence length
--max_seq_length 256
```

### Disk Space Issues
```bash
# Use streaming mode
--use_streaming
# Or limit samples
--max_train_samples 10000
```

### Slow Training with Streaming
```bash
# Increase workers
--dataloader_num_workers 8
# Or download full dataset
# Remove --use_streaming flag
```

### HuggingFace Auth Issues
```bash
# Login manually
huggingface-cli login
# Or pass token
--token "hf_your_token"
```

## Performance

Expected training time on CommonForms (full dataset):

| GPU | Batch Size | Time/Epoch | Total (3 epochs) |
|-----|-----------|------------|------------------|
| RTX 3090 | 4 | ~16 hours | ~48 hours |
| A5000 | 8 | ~12 hours | ~36 hours |
| A100 40GB | 16 | ~8 hours | ~24 hours |
| A100 80GB | 32 | ~6 hours | ~18 hours |

*Note: Streaming mode adds ~30% overhead*

## Citation

```bibtex
@article{huang2022layoutlmv3,
  title={LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  journal={arXiv preprint arXiv:2204.08387},
  year={2022}
}
```

## License

Apache 2.0 (same as original transformers-research-projects)

## Support

- Issues: [GitHub Issues](https://github.com/huggingface/transformers-research-projects/issues)
- HuggingFace Forum: https://discuss.huggingface.co/
- RunPod Discord: https://discord.gg/runpod

