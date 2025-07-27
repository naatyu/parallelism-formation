"""Pre-tokenize dataset for better loading performance."""

import argparse
import struct
from itertools import chain
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from src.utils import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        dest="data_path",
        help="Path to the data for tokenizer training.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        dest="save_path",
        help="Path where to save the trained dataset.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("parquet", data_dir=args.data_path, split="train")

    # Remove unused columns
    if dataset.column_names is not None:
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col != "text"],
        )
    else:
        logger.error("No columns found in the dataset.")

    dataset = dataset.select(range(50000))

    # Load tokenizer
    tokenizer = Tokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
    )
    eos_token_id = 2

    def tokenize(batch: dict) -> dict:
        """Tokenize a batch."""
        # Tokenize and add EOS token. We filter out empty strings.
        tokenized_texts = [
            [*tokenizer.encode(text).ids, eos_token_id]
            for text in batch["text"]
            if text
        ]
        # The output of map must be a dictionary
        return {"input_ids": tokenized_texts}

    logger.info("Tokenizing dataset...")
    # Use multiple processes for faster tokenization
    num_proc = 6
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,  # Remove old columns to save space
    )

    # Concatenate all tokens
    all_tokens = np.fromiter(
        chain.from_iterable(tokenized_dataset["input_ids"]),
        dtype=np.int32,
    )

    # Keep only full sequences
    seq_len = 1024
    num_lines = len(all_tokens) // seq_len

    # Reshape flat array
    shaped_tokens = all_tokens[: num_lines * seq_len].reshape((num_lines, seq_len))

    # Header for mmap files
    header_size = struct.calcsize("II")
    header = struct.pack("II", shaped_tokens.shape[0], shaped_tokens.shape[1])

    # Create memmap
    if not args.save_path.exists():
        args.save_path.mkdir(parents=True, exist_ok=True)
    mmap_path = Path(args.save_path) / "pretokenized.mmap"

    # Write header
    logger.info("Writing tokens to memmap file...")
    with mmap_path.open("wb") as f:
        f.write(header)
        f.seek(np.prod(shaped_tokens.shape) + header_size - 1)
        f.write(b"\0")

    # Write data
    memmap_array = np.memmap(
        mmap_path,
        dtype=np.int32,
        offset=header_size,
        mode="r+",
        shape=shaped_tokens.shape,
    )
    memmap_array[:] = shaped_tokens[:]
    memmap_array.flush()

    logger.info(f"Tokenized dataset saved at {mmap_path}")
