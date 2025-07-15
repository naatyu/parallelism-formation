"""In this first exercise we will create a custom text dataset and it's dataloader.

When implementing the dataset, be careful to the type hinting. Check also that the
collate function return a valid batch (all squences should have the same shape).
"""

from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.utils import logger

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
)  # TODO: update with local tokenizer later
tokenizer.pad_token = "</s>"


class CustomTextDataset(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.data = load_dataset("parquet", data_dir=data_dir, split="train")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]["text"]


def tokenize_collate(batch):
    return tokenizer(
        batch,
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]


if __name__ == "__main__":
    map_dataset = CustomTextDataset("data/sample/10BT")
    custom_loader = DataLoader(map_dataset, batch_size=32, collate_fn=tokenize_collate)

    # Tests
    assert len(map_dataset) == len(map_dataset.data)
    assert type(map_dataset.__getitem__(0)) is str
    assert map_dataset.__getitem__(0) == map_dataset.data[0]["text"]

    logger.info("All tests passed !")
