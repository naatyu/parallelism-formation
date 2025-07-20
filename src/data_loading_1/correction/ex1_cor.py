import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.utils import logger

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
)
tokenizer.pad_token = "</s>"  # noqa: S105


class CustomTextDataset(Dataset):
    """Simple class for our custom text dataset."""

    def __init__(self, data_dir: str) -> None:
        """Load our data."""
        self.data = load_dataset("parquet", data_dir=data_dir, split="train")

    def __len__(self) -> int:
        """Return len of our dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Return an item from our dataset."""
        return self.data[idx]["text"]


def tokenize_collate(batch: list[str]) -> torch.Tensor:
    """Collate function to tokenize batch of text."""
    return tokenizer(
        batch,
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]


if __name__ == "__main__":
    map_dataset = CustomTextDataset("data/sample/10BT")
    custom_loader = DataLoader(
        map_dataset,
        batch_size=32,
        collate_fn=tokenize_collate,
    )

    # Tests
    assert len(map_dataset) == 9672101  # noqa: PLR2004, S101
    assert type(map_dataset.__getitem__(0)) is str  # noqa: S101
    assert (  # noqa: S101
        map_dataset.__getitem__(6403)
        == (
            "What is the definition of\n[ noun ]\n(chemistry) gene that produces its "
            'characteristic phenotype only when its allele is identical\n"the recessive'
            ' gene for blue eyes"\nTo share this definition\npress "text" '
            '(Facebook, Twitter) or "link" (blog, mail) then paste'
        )
    )
    logger.info("All tests passed !")
