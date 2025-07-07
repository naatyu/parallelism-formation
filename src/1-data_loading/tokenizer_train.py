"""Train a tokenizer for our model."""

import argparse
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from src.utils import logger

TOKENIZER_PATTERN = pat_str = "|".join(  # noqa: FLY002
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ],
)


def get_training_corpus(dataset: Dataset, batch_size: int = 1000) -> Iterator[list[str]]:
    """Iterate over dataset in batches."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


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

    # Initialize BPE
    tokenizer = Tokenizer(BPE(byte_fallback=True))

    # Define pre tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=TOKENIZER_PATTERN,
        behavior="isolated",
    )

    # Define trainer
    trainer = BpeTrainer(
        vocab_size=2**16,
        show_progress=True,
        special_tokens=["<unk>", "<pad>", "<eos>", "<bos>"],
    )

    # Start of training
    tokenizer.train_from_iterator(
        get_training_corpus(dataset=dataset, batch_size=32),
        trainer,
        length=len(dataset),
    )

    # Define post-processing
    eos_token_id = tokenizer.token_to_id("<eos>")

    if eos_token_id is None:
        msg = "Special tokens <eos> not found in tokenizer vocabulary."
        raise ValueError(msg)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        pair="$A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<eos>", eos_token_id),
        ],
    )

    # Save tokenizer
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.save_path / "tokenizer"))
    logger.info(f"Saved tokenizer to {args.save_path}")

    # Test tokenizer
    pretrained_tokenizer = Tokenizer.from_file("out/pretrained_tokenizer")
    test_text = "This is a sample sentence from 2025 to tokenize."
    tokenized_input = pretrained_tokenizer(test_text, return_tensors="pt")

    logger.info(f"Original text: {test_text}")
    logger.info(f"Encoded input IDs: {tokenized_input.input_ids}")
    logger.info(f"Decoded text: {pretrained_tokenizer.decode(tokenized_input.input_ids[0])}")
