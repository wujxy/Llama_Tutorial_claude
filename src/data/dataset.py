"""
Dataset module for loading and preprocessing training data.

Usage:
    from src.data.dataset import load_jsonl, create_dataloader
    from src.tokenizer.simple_tokenizer import SimpleTokenizer

    data = load_jsonl("data/train.jsonl")
    tokenizer = SimpleTokenizer.build_vocab([f"{d['prompt']}{d['response']}" for d in data])
    dataloader = create_dataloader(data, tokenizer, max_seq_len=256, batch_size=2)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


# Instruction template for training
INSTRUCTION_TEMPLATE = """### Instruction:
{prompt}

### Response:
{response}"""


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """
    Load data from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries with 'prompt' and 'response' keys
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_instruction(prompt: str, response: str) -> str:
    """
    Format a prompt-response pair into the instruction template.

    Args:
        prompt: Input prompt
        response: Expected response

    Returns:
        Formatted string
    """
    return INSTRUCTION_TEMPLATE.format(prompt=prompt, response=response)


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data with tokenization.

    This dataset handles:
    - Loading prompt-response pairs
    - Formatting with instruction template
    - Tokenization
    - Creating causal LM labels (right-shifted, with padding masked)
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_seq_len: int = 256,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data: List of dictionaries with 'prompt' and 'response' keys
            tokenizer: SimpleTokenizer instance
            max_seq_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.

        Args:
            idx: Index

        Returns:
            Dictionary with:
                - input_ids: Token IDs (shape: [seq_len])
                - labels: Label IDs for causal LM (shape: [seq_len])
                - attention_mask: Mask for padding (1=real, 0=pad)
        """
        item = self.data[idx]

        # Format the instruction
        text = format_instruction(item['prompt'], item['response'])

        # Tokenize with BOS and EOS
        input_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        # Truncate if necessary
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # Create labels (right-shifted for causal LM)
        # For causal LM, label[i] predicts input_ids[i]
        # But we mask padding tokens with -100
        labels = input_ids.copy()

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.PAD_ID
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len  # Mask padding in labels
            attention_mask = attention_mask + [0] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


def create_dataloader(
    data: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int = 256,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    Args:
        data: List of dictionaries with 'prompt' and 'response' keys
        tokenizer: SimpleTokenizer instance
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = TextDataset(data, tokenizer, max_seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Set to True for CUDA
    )
