import math
import struct
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.utils import logger


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
        raise NotImplementedError

    def __iter__(self) -> Generator:
        """Yield tokenized sequences."""
        # Get worker information.
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading: the worker processes the entire dataset.
            # Implement the `start_index` and `end_index` for a single worker
            start_index = NotImplemented
            end_index = NotImplemented
        else:
            # Multi-process loading: each worker gets a specific chunk.
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Implement the `start_index` and `end_index` for multiple workers
            # Notice that worker_id and num_workers are available :)
            # Tip: use math.ceil for upper rounding
            start_index = NotImplemented
            end_index = NotImplemented

        # Iterate through the assigned chunk of data.
        for i in range(start_index, end_index):
            # Fetch the sequence. Don't forget to copy the data for it to be available
            sequence = NotImplemented

            # Convert to a tensor and yield data with long data type.
            yield NotImplementedError


if __name__ == "__main__":
    mmap_file_path = Path("data/pretokenized_dataset/pretokenized.mmap")
    iterable_dataset = MemmapIterableDataset(mmap_path=mmap_file_path)

    dataloader = DataLoader(
        iterable_dataset,
        batch_size=8,
        num_workers=0,
    )
    sample = next(iter(dataloader))
    multiprocess_dataloader = DataLoader(
        iterable_dataset,
        batch_size=8,
        num_workers=4,
    )
    multiprocess_sample = next(iter(multiprocess_dataloader))

    # Tests
    assert len(iterable_dataset) == 57044  # noqa: PLR2004, S101
    assert multiprocess_sample.dtype == torch.int64  # noqa: S101
    assert sample.dtype == torch.int64  # noqa: S101

    logger.info("All tests passed !")
