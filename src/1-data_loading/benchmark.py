"""Benchmark of different implementations mades."""

import argparse
import statistics
import time
from pathlib import Path

import correction.ex1_cor as ex1
from torch.utils.data import DataLoader

from src.utils import logger


def benchmark(loader):
    step_times = []
    for i in range(500):
        step_start_time = time.time()
        next(loader)
        step_end_time = time.time()
        elapsed_time = step_end_time - step_start_time

        if i == 0:
            step_first_time = elapsed_time
        step_times.append(elapsed_time)
    print(  # noqa: T201
        "\n------------------------------------"
        f"\nTime to first step: {step_first_time * 1e3:.2f} ms"
        f"\nMean time/batch: {statistics.mean(step_times) * 1e3:.3f} ms"
        f"\nMin time: {min(step_times) * 1e3:.3f} ms"
        f"\nMax time: {max(step_times) * 1e3:.3f} ms"
        "\n------------------------------------",
    )


def benchmark_gpu_transfert(loader):
    step_times = []
    for i in range(500):
        batch = next(loader)
        step_start_time = time.time()
        batch.to("cuda")
        step_end_time = time.time()
        elapsed_time = step_end_time - step_start_time

        if i == 0:
            step_first_time = elapsed_time
        step_times.append(elapsed_time)
    print(  # noqa: T201
        "\n------------------------------------"
        f"\nTime to first step: {step_first_time * 1e3:.2f} ms"
        f"\nMean time/batch: {statistics.mean(step_times) * 1e3:.3f} ms"
        f"\nMin time: {min(step_times) * 1e3:.3f} ms"
        f"\nMax time: {max(step_times) * 1e3:.3f} ms"
        "\n------------------------------------",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default="data/sample/10BT",
        dest="data_path",
        help="Path to the data for tokenizer training.",
    )

    args = parser.parse_args()

    # --- Bench simple map dataset --- #
    simple_map_dataset = ex1.CustomTextDataset(args.data_path)
    custom_loader = iter(
        DataLoader(simple_map_dataset, batch_size=32, collate_fn=ex1.tokenize_collate),
    )
    logger.info("Simple map dataset")
    benchmark(custom_loader)

    # --- Bench multi-worker map dataset --- #
    custom_multi_loader = iter(
        DataLoader(
            simple_map_dataset,
            batch_size=32,
            collate_fn=ex1.tokenize_collate,
            num_workers=6,
        ),
    )
    logger.info("Multi worker dataset")
    benchmark(custom_multi_loader)

    # --- Bench pin-memory dataset --- #
    custom_pin_loader = iter(
        DataLoader(
            simple_map_dataset,
            batch_size=32,
            collate_fn=ex1.tokenize_collate,
            pin_memory=True,
        ),
    )
    logger.info("Without pin memory")
    benchmark_gpu_transfert(custom_loader)
    logger.info("With pin memory")
    benchmark_gpu_transfert(custom_pin_loader)
