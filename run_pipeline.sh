#!/bin/bash

# Quick test script for LLaMA from scratch + LoRA pipeline
# This script runs the complete training and evaluation pipeline

set -e  # Exit on error

echo "=========================================="
echo "LLaMA from Scratch + LoRA Quick Test"
echo "=========================================="

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "[0/6] Cleaning old outputs..."
rm -rf outputs/base_train/* outputs/lora_pretrain/* outputs/eval/* 2>/dev/null || true

# Clean old outputs to avoid permission issues
echo ""
echo "[0/6] Cleaning old outputs..."
rm -rf outputs/base_train/* outputs/lora_pretrain/* outputs/eval/* 2>/dev/null || true

# Activate virtual environment
echo ""
echo "[1/6] Activating virtual environment..."
source /home/NagaiYoru/LLM_Tutorial/llm_py_venv/bin/activate

# Check dependencies
echo ""
echo "[2/6] Checking dependencies..."
python -c "import torch, yaml, tqdm" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q torch pyyaml tqdm numpy
}

# Step 1: Base training
echo ""
echo "[3/6] Training base model..."
echo "----------------------------"
python -m src.train_base --config configs/train.yaml

# Step 2: LoRA training
echo ""
echo "[4/6] Training LoRA adapters..."
echo "-------------------------------"
python -m src.train_lora --config configs/lora_pretrain.yaml

# Step 3: Evaluate base model
echo ""
echo "[5/6] Evaluating base model..."
echo "------------------------------"
python -m src.eval --config configs/eval.yaml

# Backup original eval config
cp configs/eval.yaml configs/eval.yaml.bak

# Step 4: Evaluate base + LoRA
echo ""
echo "[6/6] Evaluating base + LoRA model..."
echo "-------------------------------------"
# Enable LoRA in eval config
sed -i 's/lora_ckpt_dir: ""/lora_ckpt_dir: "outputs\/lora_pretrain"/' configs/eval.yaml
python -m src.eval --config configs/eval.yaml

# Restore original eval config
mv configs/eval.yaml.bak configs/eval.yaml

# Summary
echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Base model:     outputs/base_train/"
echo "  - LoRA adapters:  outputs/lora_pretrain/"
echo "  - Evaluation:     outputs/eval/"
echo ""
echo "Check results:"
echo "  - cat outputs/base_train/train_log.txt"
echo "  - cat outputs/lora_pretrain/train_log.txt"
echo "  - cat outputs/eval/metrics.json"
echo "  - cat outputs/eval/generations.txt"
echo ""
