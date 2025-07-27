import math
import struct
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class MemmapIterableDataset(IterableDataset):
    """IterableDataset for reading memory-map file.

    It is optimized for sequential reading and correctly handles data
    partitioning for multi-worker data loading.

    Args:
        mmap_path (Path | str): Path to the .mmap file.

    """

    def __init__(self, mmap_path: Path | str) -> None:
        """Initialize."""
        super().__init__()
        self.mmap_path = Path(mmap_path)

        # Check if mmap exist
        if not self.mmap_path.is_file():
            msg = f"Memmap file not found at: {self.mmap_path}"
            raise FileNotFoundError(msg)

        # Read header
        header_size = struct.calcsize("II")
        with self.mmap_path.open("rb") as f:
            header_bytes = f.read(header_size)
        self.num_seq, self.seq_len = struct.unpack("II", header_bytes)

        # Get data, not placed in RAM
        self.data = np.memmap(
            self.mmap_path,
            dtype=np.int32,
            mode="r",
            offset=header_size,
            shape=(self.num_seq, self.seq_len),
        )

    def __len__(self) -> int:
        """Return len of our dataset."""
        return self.num_seq

    def __iter__(self) -> Generator:
        """Yield tokenized sequences."""
        # Get worker information.
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading: the worker processes the entire dataset.
            start_index = 0
            end_index = self.num_seq
        else:
            # Multi-process loading: each worker gets a specific chunk.
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Calculate the chunk size for each worker.
            per_worker = math.ceil(self.num_seq / num_workers)
            start_index = worker_id * per_worker
            end_index = min(start_index + per_worker, self.num_seq)

        # Iterate through the assigned chunk of data.
        for i in range(start_index, end_index):
            # Fetch the sequence. Using .copy() is important for multiprocessing
            # to ensure each worker gets its own copy of the data slice.
            sequence = self.data[i].copy()

            # Convert to a tensor and yield in a dictionary format.
            yield torch.from_numpy(sequence).long()


# --- Example Usage ---
if __name__ == "__main__":
    # Assume the pre-tokenization script has been run and created this file.
    MMAP_FILE_PATH = Path("data/pretokenized_dataset/pretokenized.mmap")

    # 1. Create an instance of the dataset
    if MMAP_FILE_PATH.exists():
        print(f"Loading dataset from {MMAP_FILE_PATH}...")
        iterable_dataset = MemmapIterableDataset(mmap_path=MMAP_FILE_PATH)

        # 2. Use it with a DataLoader
        # Set num_workers > 0 to see the multi-worker partitioning in action.
        dataloader = DataLoader(
            iterable_dataset,
            batch_size=8,
            num_workers=4,
            pin_memory=True,
        )

        print(f"\nCreated DataLoader with {dataloader.num_workers} workers.")
        print(f"Total sequences in dataset: {len(iterable_dataset)}")

        # 3. Iterate through a few batches to test
        print("\nFetching a few sample batches...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}:")
            print("  input_ids shape:", batch.shape)
            print("  input_ids dtype:", batch.dtype)

            if i >= 2:  # Stop after fetching 3 batches for this demo
                break

        print("\n✅ IterableDataset with multi-worker support is working correctly.")

    else:
        print(f"❌ Error: Memmap file not found at '{MMAP_FILE_PATH}'.")
        print("Please run the pre-tokenization script first.")
