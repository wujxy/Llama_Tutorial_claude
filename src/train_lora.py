"""
LoRA training script - fine-tunes with frozen base model.

This script loads a pre-trained base model and injects LoRA adapters,
then trains only the LoRA parameters while keeping the base model frozen.

Usage:
    python -m src.train_lora --config configs/lora_pretrain.yaml

Input:
    - Base model checkpoint from outputs/base_train/model.pt
    - LoRA training data JSONL file
    - LoRA hyperparameters (rank, alpha, dropout, target modules)

Output:
    - outputs/lora_pretrain/lora.pt: LoRA weights only
    - outputs/lora_pretrain/lora_config.json: LoRA configuration
    - outputs/lora_pretrain/train_log.txt: Training log
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataloader, load_jsonl
from src.lora.lora import (
    LoRALinear,
    apply_lora,
    freeze_base_model,
    get_lora_parameters,
    save_lora_weights,
)
from src.llama.model import ModelArgs, Transformer
from src.tokenizer.simple_tokenizer import SimpleTokenizer
from src.utils.io import ensure_dir, load_checkpoint, save_json
from src.utils.logging import Logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train LoRA adapters")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_pretrain.yaml",
        help="Path to LoRA training config file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: Logger,
    log_every: int = 10,
) -> float:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        log_every: Log frequency

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

        # Log
        if (step + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            logger.log(f"Epoch {epoch}, Step {step + 1}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

    return total_loss / num_batches


def count_parameters(model: Transformer) -> tuple[int, int]:
    """
    Count total and trainable parameters.

    Args:
        model: The model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = ensure_dir(config['output_dir'])
    logger = Logger(str(output_dir / "train_log.txt"))

    logger.log("=" * 50)
    logger.log("Starting LoRA training")
    logger.log("=" * 50)

    # Load base model checkpoint
    base_ckpt_dir = Path(config['base_ckpt_dir'])
    model_path = base_ckpt_dir / "model.pt"
    config_path = base_ckpt_dir / "config.json"
    tokenizer_path = base_ckpt_dir / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Base model checkpoint not found at {model_path}. "
            "Please run base training first: python -m src.train_base --config configs/train.yaml"
        )

    logger.log(f"Loading base model from {model_path}")

    # Load model config
    import json
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    model_args_dict = saved_config['model_args']

    # Create model
    model_args = ModelArgs(
        vocab_size=model_args_dict['vocab_size'],
        dim=model_args_dict['dim'],
        n_layers=model_args_dict['n_layers'],
        n_heads=model_args_dict['n_heads'],
        n_kv_heads=model_args_dict['n_kv_heads'],
        hidden_dim=model_args_dict['hidden_dim'],
        max_seq_len=model_args_dict['max_seq_len'],
        dropout=model_args_dict.get('dropout', 0.0),
    )

    model = Transformer(model_args)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    logger.log("Base model loaded successfully")

    # Load tokenizer
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    logger.log(f"Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    # Apply LoRA
    logger.log("Applying LoRA to model...")

    lora_config = config['lora']
    target_modules = lora_config.get('target_modules', [])

    # Default target modules for LLaMA
    if not target_modules:
        target_modules = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']

    replaced_count = apply_lora(
        model,
        r=lora_config['r'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=target_modules,
    )

    logger.log(f"LoRA applied to {replaced_count} modules")
    logger.log(f"Target modules: {target_modules}")

    # Freeze base model parameters
    freeze_base_model(model)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    lora_params = len(get_lora_parameters(model))

    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable (LoRA) parameters: {trainable_params:,}")
    logger.log(f"LoRA parameters: {lora_params}")
    logger.log(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Load LoRA training data
    logger.log(f"Loading LoRA training data from {config['data_path']}")
    data = load_jsonl(config['data_path'])
    logger.log(f"Loaded {len(data)} samples")

    # Create dataloader
    max_seq_len = config['max_seq_len']
    batch_size = config['train']['batch_size']

    dataloader = create_dataloader(
        data,
        tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
    )

    # Create optimizer (only for LoRA parameters)
    logger.log("Creating optimizer for LoRA parameters...")
    lora_params_list = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(
        lora_params_list,
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )

    logger.log(f"Optimizer created with {len(list(optimizer.param_groups[0]['params']))} parameter groups")

    # Training loop
    logger.log("Starting LoRA training...")

    max_steps = config['train']['max_steps']
    log_every = config['train']['log_every']

    global_step = 0
    epoch = 0

    while global_step < max_steps:
        avg_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            device,
            epoch,
            logger,
            log_every=log_every,
        )

        logger.log(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        steps_per_epoch = len(dataloader)
        global_step += steps_per_epoch
        epoch += 1

        if global_step >= max_steps:
            break

    # Save LoRA weights
    logger.log("Training completed. Saving LoRA weights...")

    lora_weights_path = output_dir / "lora.pt"
    save_lora_weights(model, str(lora_weights_path))
    logger.log(f"LoRA weights saved to {lora_weights_path}")

    # Save LoRA config
    lora_config_to_save = {
        'r': lora_config['r'],
        'alpha': lora_config['alpha'],
        'dropout': lora_config['dropout'],
        'target_modules': target_modules,
        'base_model_dir': str(base_ckpt_dir),
    }

    lora_config_path = output_dir / "lora_config.json"
    save_json(lora_config_to_save, lora_config_path)
    logger.log(f"LoRA config saved to {lora_config_path}")

    logger.log("=" * 50)
    logger.log("LoRA training completed successfully!")
    logger.log("=" * 50)


if __name__ == "__main__":
    main()
