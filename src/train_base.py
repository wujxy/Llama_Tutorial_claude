"""
Base model training script - trains LLaMA from scratch.

This script trains a LLaMA model from random initialization on the provided data.
No pre-trained weights are used - this is true "from scratch" training.

Usage:
    python -m src.train_base --config configs/train.yaml

Input:
    - Training data JSONL file (specified in config)
    - Model hyperparameters (from config)

Output:
    - outputs/base_train/model.pt: Trained model checkpoint
    - outputs/base_train/tokenizer.json: Tokenizer vocabulary
    - outputs/base_train/config.json: Model configuration
    - outputs/base_train/train_log.txt: Training log
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataloader, load_jsonl
from src.llama.model import ModelArgs, Transformer
from src.tokenizer.simple_tokenizer import SimpleTokenizer
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.logging import Logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train LLaMA model from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config file",
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
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross-entropy
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute cross-entropy loss (ignore padding tokens with -100)
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
    logger.log("Starting base model training from scratch")
    logger.log("=" * 50)

    # Load training data
    logger.log(f"Loading training data from {config['data_path']}")
    data = load_jsonl(config['data_path'])
    logger.log(f"Loaded {len(data)} samples")

    # Build tokenizer from all text data
    logger.log("Building tokenizer...")
    all_texts = [f"{d['prompt']}{d['response']}" for d in data]
    tokenizer = SimpleTokenizer.build_vocab(all_texts)
    logger.log(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Create dataloader
    max_seq_len = config['max_seq_len']
    batch_size = config['train']['batch_size']

    logger.log(f"Creating dataloader (max_seq_len={max_seq_len}, batch_size={batch_size})")
    dataloader = create_dataloader(
        data,
        tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
    )

    # Create model
    logger.log("Creating model...")

    # Get model config from YAML
    model_config = config['model']
    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        dim=model_config['dim'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        n_kv_heads=model_config.get('n_kv_heads', model_config['n_heads']),
        hidden_dim=model_config['hidden_dim'],
        max_seq_len=max_seq_len,
        dropout=model_config.get('dropout', 0.0),
    )

    model = Transformer(model_args)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Model created with {total_params:,} total parameters")
    logger.log(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    logger.log("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )

    # Training loop
    logger.log("Starting training...")

    max_steps = config['train']['max_steps']
    log_every = config['train']['log_every']

    # Since we have a small dataset, we'll train for multiple epochs
    # but limit total steps
    global_step = 0
    epoch = 0

    while global_step < max_steps:
        # Train for one epoch
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

        # Update step count
        steps_per_epoch = len(dataloader)
        global_step += steps_per_epoch

        epoch += 1

        # Break if we've reached max steps
        if global_step >= max_steps:
            break

    # Save checkpoint
    logger.log("Training completed. Saving checkpoint...")

    # Save model
    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.log(f"Model saved to {checkpoint_path}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.log(f"Tokenizer saved to {tokenizer_path}")

    # Save config
    config_to_save = {
        'model_args': {
            'vocab_size': tokenizer.vocab_size,
            'dim': model_args.dim,
            'n_layers': model_args.n_layers,
            'n_heads': model_args.n_heads,
            'n_kv_heads': model_args.n_kv_heads,
            'hidden_dim': model_args.hidden_dim,
            'max_seq_len': model_args.max_seq_len,
            'dropout': model_args.dropout,
        },
        'training_config': config,
    }

    config_path = output_dir / "config.json"
    save_json(config_to_save, config_path)
    logger.log(f"Config saved to {config_path}")

    logger.log("=" * 50)
    logger.log("Base model training completed successfully!")
    logger.log("=" * 50)


if __name__ == "__main__":
    main()
