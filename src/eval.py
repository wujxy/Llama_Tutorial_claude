"""
Evaluation script for base and base+LoRA models.

This script evaluates a trained model on test data, computing:
- Average loss
- Perplexity (exp(loss))
- Text generation samples

Usage:
    python -m src.eval --config configs/eval.yaml

Input:
    - Base model checkpoint
    - Optional LoRA checkpoint
    - Evaluation data JSONL file

Output:
    - outputs/eval/metrics.json: Loss and perplexity
    - outputs/eval/generations.txt: Generated text samples
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import INSTRUCTION_TEMPLATE, load_jsonl
from src.lora.lora import apply_lora, load_lora_weights
from src.llama.model import ModelArgs, Transformer
from src.tokenizer.simple_tokenizer import SimpleTokenizer
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.logging import Logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation config file",
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


def evaluate_loss(
    model: Transformer,
    data: list,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    max_seq_len: int = 256,
    max_eval_samples: int = None,
) -> tuple[float, int]:
    """
    Evaluate model loss on data.

    Args:
        model: The model to evaluate
        data: List of dictionaries with 'prompt' and 'response'
        tokenizer: Tokenizer
        device: Device to evaluate on
        max_seq_len: Maximum sequence length
        max_eval_samples: Maximum number of samples to evaluate

    Returns:
        Tuple of (average_loss, num_samples_evaluated)
    """
    model.eval()

    if max_eval_samples is not None:
        data = data[:max_eval_samples]

    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating loss"):
            # Format text
            text = INSTRUCTION_TEMPLATE.format(
                prompt=item['prompt'],
                response=item['response'],
            )

            # Tokenize
            input_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

            # Truncate if needed
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]

            # Convert to tensor
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = input_ids[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            total_loss += loss.item()
            num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss, num_samples


@torch.inference_mode()
def generate_text(
    model: Transformer,
    prompt: str,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = None,
) -> str:
    """
    Generate text using greedy decoding.

    Args:
        model: The model
        prompt: Input prompt
        tokenizer: Tokenizer
        device: Device to generate on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no scaling)
        top_k: Top-k sampling (None = greedy)

    Returns:
        Generated text
    """
    model.eval()

    # Format prompt (without response)
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Tokenize
    input_ids = tokenizer.encode(text, add_bos=True, add_eos=False)

    # Convert to tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Generate
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass
        logits = model(generated_ids)

        # Get next token logits
        next_token_logits = logits[0, -1, :] / temperature

        # Apply top-k sampling if specified
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[-1]] = float('-inf')

        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        # Stop if we generate EOS
        if next_token.item() == tokenizer.EOS_ID:
            break

    # Decode
    generated_ids = generated_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special=True)

    # Extract just the response part
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[-1].strip()
    else:
        response = generated_text

    return response


def main() -> None:
    """
    Main evaluation function.
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
    logger = Logger(str(output_dir / "eval_log.txt"))

    logger.log("=" * 50)
    logger.log("Starting model evaluation")
    logger.log("=" * 50)

    # Load base model checkpoint
    base_ckpt_dir = Path(config['base_ckpt_dir'])
    model_path = base_ckpt_dir / "model.pt"
    config_path = base_ckpt_dir / "config.json"
    tokenizer_path = base_ckpt_dir / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            "Please run training first."
        )

    logger.log(f"Loading model from {model_path}")

    # Load model config
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
        dropout=0.0,  # No dropout during evaluation
    )

    model = Transformer(model_args)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    logger.log("Base model loaded successfully")

    # Load LoRA if specified
    lora_ckpt_dir = config.get('lora_ckpt_dir')
    if lora_ckpt_dir:
        logger.log(f"Loading LoRA weights from {lora_ckpt_dir}")

        lora_path = Path(lora_ckpt_dir) / "lora.pt"
        lora_config_path = Path(lora_ckpt_dir) / "lora_config.json"

        if not lora_path.exists():
            logger.log(f"Warning: LoRA checkpoint not found at {lora_path}")
            logger.log("Proceeding with base model only")
        else:
            # Load LoRA config
            with open(lora_config_path, 'r') as f:
                lora_config = json.load(f)

            # Apply LoRA to model
            apply_lora(
                model,
                r=lora_config['r'],
                alpha=lora_config['alpha'],
                dropout=0.0,  # No dropout during evaluation
                target_modules=lora_config['target_modules'],
            )

            # Load LoRA weights
            load_lora_weights(model, str(lora_path))
            logger.log("LoRA weights loaded and applied")

    # Load tokenizer
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Load evaluation data
    logger.log(f"Loading evaluation data from {config['data_path']}")
    eval_data = load_jsonl(config['data_path'])
    logger.log(f"Loaded {len(eval_data)} samples")

    # Evaluate loss and perplexity
    logger.log("Computing loss and perplexity...")

    batch_size = config['eval']['batch_size']
    max_eval_samples = config['eval'].get('max_eval_samples', None)

    avg_loss, num_samples = evaluate_loss(
        model,
        eval_data,
        tokenizer,
        device,
        max_seq_len=config['max_seq_len'],
        max_eval_samples=max_eval_samples,
    )

    perplexity = float(f"inf") if avg_loss == float('inf') else float(f"{torch.exp(torch.tensor(avg_loss)):.4f}")

    logger.log(f"Evaluation completed on {num_samples} samples")
    logger.log(f"Average loss: {avg_loss:.4f}")
    logger.log(f"Perplexity: {perplexity:.4f}")

    # Save metrics
    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_samples': num_samples,
        'model_type': 'base+lora' if lora_ckpt_dir else 'base',
    }

    metrics_path = output_dir / "metrics.json"
    save_json(metrics, metrics_path)
    logger.log(f"Metrics saved to {metrics_path}")

    # Generate text samples
    logger.log("Generating text samples...")

    max_new_tokens = config['eval']['gen_max_new_tokens']
    num_gen_samples = min(2, len(eval_data))  # Generate for first 2 samples

    generations = []
    for i in range(num_gen_samples):
        item = eval_data[i]
        prompt = item['prompt']
        expected_response = item['response']

        logger.log(f"\n--- Sample {i + 1} ---")
        logger.log(f"Prompt: {prompt}")

        generated = generate_text(
            model,
            prompt,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
        )

        logger.log(f"Expected: {expected_response}")
        logger.log(f"Generated: {generated}")

        generations.append({
            'sample_id': i + 1,
            'prompt': prompt,
            'expected': expected_response,
            'generated': generated,
        })

    # Save generations
    generations_path = output_dir / "generations.txt"
    with open(generations_path, 'w', encoding='utf-8') as f:
        for gen in generations:
            f.write(f"=== Sample {gen['sample_id']} ===\n")
            f.write(f"Prompt: {gen['prompt']}\n")
            f.write(f"Expected: {gen['expected']}\n")
            f.write(f"Generated: {gen['generated']}\n")
            f.write("\n")

    logger.log(f"Generations saved to {generations_path}")

    logger.log("=" * 50)
    logger.log("Evaluation completed successfully!")
    logger.log("=" * 50)


if __name__ == "__main__":
    main()
