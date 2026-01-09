# LLaMA from Scratch + LoRA: Hello World Pipeline

A minimal, self-contained implementation of LLaMA-style transformer trained from scratch with LoRA fine-tuning support.

## Features

- **Zero external dependencies**: No pre-trained weights or HuggingFace models
- **Character-level tokenizer**: Simple, self-contained tokenization
- **LLaMA architecture**: RMSNorm, RoPE, SwiGLU, causal attention
- **LoRA support**: Efficient fine-tuning with frozen base model
- **Complete pipeline**: Training → LoRA fine-tuning → Evaluation

## Project Structure

```
.
├── configs/                  # Configuration files
│   ├── train.yaml           # Base model training config
│   ├── lora_pretrain.yaml   # LoRA training config
│   └── eval.yaml            # Evaluation config
├── data/                     # Training data
│   ├── train.jsonl          # Base training data (10 samples)
│   ├── lora.jsonl           # LoRA training data (10 samples)
│   └── eval.jsonl           # Evaluation data (10 samples)
├── outputs/                  # Output directory
│   ├── base_train/          # Base model checkpoints
│   ├── lora_pretrain/       # LoRA adapter checkpoints
│   └── eval/                # Evaluation results
└── src/                      # Source code
    ├── llama/               # LLaMA model implementation
    ├── lora/                # LoRA implementation
    ├── tokenizer/           # Character-level tokenizer
    ├── data/                # Dataset loader
    ├── utils/               # Utility functions
    ├── train_base.py        # Base training script
    ├── train_lora.py        # LoRA training script
    └── eval.py              # Evaluation script
```

## Requirements

- Python 3.10
- PyTorch
- NumPy
- PyYAML
- tqdm

## Installation

1. Clone the repository:
```bash
cd /home/NagaiYoru/LLM_Tutorial/llama_claude
```

2. Create and activate virtual environment (already created):
```bash
source /home/NagaiYoru/LLM_Tutorial/llm_py_venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train Base Model

Train a LLaMA model from scratch on random data:

```bash
python -m src.train_base --config configs/train.yaml
```

**Output:**
- `outputs/base_train/model.pt` - Trained model checkpoint
- `outputs/base_train/tokenizer.json` - Tokenizer vocabulary
- `outputs/base_train/config.json` - Model configuration
- `outputs/base_train/train_log.txt` - Training log

**Expected output:**
```
Using device: cuda (or cpu)
Loading training data from data/train.jsonl
Loaded 10 samples
Building tokenizer...
Tokenizer vocabulary size: ~50
Creating model...
Model created with ~100,000 total parameters
Starting training...
Epoch 0, Step 10, Loss: ~3.0
...
Training completed. Saving checkpoint...
```

### Step 2: Train LoRA Adapters

Fine-tune with LoRA (freezes base model, trains only LoRA parameters):

```bash
python -m src.train_lora --config configs/lora_pretrain.yaml
```

**Output:**
- `outputs/lora_pretrain/lora.pt` - LoRA weights only
- `outputs/lora_pretrain/lora_config.json` - LoRA configuration
- `outputs/lora_pretrain/train_log.txt` - Training log

**Expected output:**
```
Using device: cuda (or cpu)
Loading base model from outputs/base_train/model.pt
Base model loaded successfully
Applying LoRA to model...
LoRA applied to 14 modules
Total parameters: 100,000
Trainable (LoRA) parameters: ~5,000
Percentage trainable: ~5%
Starting LoRA training...
...
LoRA training completed successfully!
```

### Step 3: Evaluate Base Model

Evaluate the base model:

```bash
python -m src.eval --config configs/eval.yaml
```

**Output:**
- `outputs/eval/metrics.json` - Loss and perplexity
- `outputs/eval/generations.txt` - Generated text samples

### Step 4: Evaluate Base + LoRA Model

To evaluate with LoRA, edit `configs/eval.yaml`:

```yaml
lora_ckpt_dir: "outputs/lora_pretrain"  # Uncomment this line
```

Then run:

```bash
python -m src.eval --config configs/eval.yaml
```

## Model Architecture

The model implements the LLaMA architecture with the following components:

- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Position Embeddings
- **Multi-head Attention**: With causal masking
- **SwiGLU**: Activation function (SiLU variant)
- **Pre-normalization**: Normalization before sub-layers

Default configuration (small for CPU training):
- Dimension: 128
- Layers: 2
- Attention heads: 4
- Hidden dimension: 256
- Max sequence length: 256

## Data Format

Data is stored in JSONL format with one sample per line:

```json
{"prompt": "Add two numbers: 3 + 5", "response": "8"}
{"prompt": "Reverse string: abcde", "response": "edcba"}
```

The training format uses an instruction template:

```
### Instruction:
{prompt}

### Response:
{response}
```

## LoRA Configuration

LoRA is applied to the following modules:
- Attention: `wq`, `wk`, `wv`, `wo`
- Feed-forward: `w1`, `w2`, `w3`

Default LoRA hyperparameters:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.0

This results in ~5% of the total parameters being trainable.

## Troubleshooting

**Issue**: "Base model checkpoint not found"
- **Solution**: Run base training first: `python -m src.train_base --config configs/train.yaml`

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in config file or use CPU (it will be slower)

**Issue**: Training is slow on CPU
- **Solution**: Reduce `max_steps` or model size (`dim`, `n_layers`) in config

**Issue**: Generated text is poor quality
- **Solution**: This is expected! The model is tiny and trained on 10 samples. This is a Hello World demo, not a production model.

## Notes

- This is a **Hello World** implementation for educational purposes
- The model size is tiny (128 dimensions, 2 layers) and trained on only 10 samples
- Don't expect meaningful language generation - this demonstrates the pipeline, not performance
- For production use, you would need: larger model, more data, longer training

## License

This project is for educational purposes.
