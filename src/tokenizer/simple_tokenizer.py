"""
Simple character-level tokenizer for training from scratch.

This tokenizer builds a vocabulary from all unique characters in the training data
and supports encoding/decoding text. It's designed for "Hello World" style
demonstrations where we want minimal dependencies and clear code.

Usage:
    from src.tokenizer.simple_tokenizer import SimpleTokenizer

    # Build vocab from texts
    tokenizer = SimpleTokenizer.build_vocab(texts)

    # Encode and decode
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

    # Save and load
    tokenizer.save("vocab.json")
    tokenizer = SimpleTokenizer.load("vocab.json")
"""

import json
from pathlib import Path
from typing import Dict, List, Set


class SimpleTokenizer:
    """
    Character-level tokenizer with special tokens.

    Special tokens:
        <pad>: Padding token (id=0)
        <bos>: Beginning of sequence (id=1)
        <eos>: End of sequence (id=2)
        <unk>: Unknown token (id=3)
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Special token IDs (fixed for consistency)
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab: Dict[str, int]) -> None:
        """
        Initialize tokenizer with a vocabulary.

        Args:
            vocab: Dictionary mapping characters to IDs
        """
        self.vocab = vocab
        self.inverse_vocab = {idx: char for char, idx in vocab.items()}
        self.vocab_size = len(vocab)

    @classmethod
    def build_vocab(cls, texts: List[str], max_vocab_size: int = 1000) -> 'SimpleTokenizer':
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of strings to build vocab from
            max_vocab_size: Maximum vocabulary size (for safety)

        Returns:
            SimpleTokenizer instance
        """
        # Start with special tokens
        vocab: Dict[str, int] = {
            cls.PAD_TOKEN: cls.PAD_ID,
            cls.BOS_TOKEN: cls.BOS_ID,
            cls.EOS_TOKEN: cls.EOS_ID,
            cls.UNK_TOKEN: cls.UNK_ID,
        }

        # Collect unique characters
        char_set: Set[str] = set()
        for text in texts:
            char_set.update(text)

        # Add characters to vocab (sorted for consistency)
        for char in sorted(char_set):
            if char not in vocab:
                vocab[char] = len(vocab)
                if len(vocab) >= max_vocab_size:
                    break

        return cls(vocab)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input string
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            List of token IDs
        """
        ids = []

        if add_bos:
            ids.append(self.BOS_ID)

        for char in text:
            ids.append(self.vocab.get(char, self.UNK_ID))

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special: Whether to skip special tokens

        Returns:
            Decoded string
        """
        special_ids = {self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID}

        chars = []
        for idx in ids:
            if skip_special and idx in special_ids:
                continue
            char = self.inverse_vocab.get(idx, self.UNK_TOKEN)
            chars.append(char)

        return ''.join(chars)

    def save(self, path: str) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """
        Load vocabulary from JSON file.

        Args:
            path: Input file path

        Returns:
            SimpleTokenizer instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        return cls(vocab)
