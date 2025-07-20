# ruff: noqa: T201
"""Benchmark of different DataLoader implementations."""

import argparse
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.utils import logger

from . import ex1


def run_benchmark(  # noqa: PLR0913
    description: str,
    loader: Iterator,
    action: Callable[[Any], None] | None,
    num_steps: int = 500,
    warmup_steps: int = 10,
    *,
    use_cuda_sync: bool = False,
) -> None:
    """Run a benchmark for a given action and data loader.

    Args:
        description (str): A description of the benchmark being run.
        loader (Iterator): The data loader iterator.
        action (Callable[[Any], None] | None): The function to benchmark. If None,
            it times the data loading (`next(loader)`). Otherwise, it times the
            action performed on the loaded batch.
        num_steps (int): The number of steps to measure.
        warmup_steps (int): The number of initial steps to ignore for stable results.
        use_cuda_sync (bool): Whether to use torch.cuda.synchronize() for accurate
            GPU timing.

    """
    # Warm-up phase to exclude initial setup costs (e.g., worker spin-up)
    for _ in range(warmup_steps):
        batch = next(loader)
        if action:
            action(batch)

    timings = []
    for _ in range(num_steps):
        # Case 1: Benchmark only the data loading time
        if action is None:
            start_time = time.perf_counter()
            _ = next(loader)
            end_time = time.perf_counter()
        # Case 2: Benchmark an action on the data (e.g., GPU transfer)
        else:
            batch = next(loader)
            if use_cuda_sync:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            action(batch)

            if use_cuda_sync:
                torch.cuda.synchronize()
            end_time = time.perf_counter()

        timings.append(end_time - start_time)

    # Report results
    print(f"\n--- {description} ---")
    if not timings:
        print("No timings recorded.")
        return

    mean_ms = statistics.mean(timings) * 1e3
    stdev_ms = statistics.stdev(timings) * 1e3 if len(timings) > 1 else 0.0

    print(f"Mean time/batch: {mean_ms:.3f} ms")
    print(f"Std Dev:         {stdev_ms:.3f} ms")
    print(f"Min time:        {min(timings) * 1e3:.3f} ms")
    print(f"Max time:        {max(timings) * 1e3:.3f} ms")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark different DataLoader implementations.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default="data/sample/10BT",
        help="Path to the data.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of warm-up iterations.",
    )

    args = parser.parse_args()

    # --- Setup ---
    base_dataset = ex1.CustomTextDataset(args.data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA not available. Skipping GPU-related benchmarks.")

    # --- 1. Benchmark DataLoader iteration speed ---
    logger.info("--- Benchmarking DataLoader iteration speed ---")
    cpu_configs = [
        {
            "description": "CPU Iteration: 1 worker",
            "loader_params": {"num_workers": 0},
        },
        {
            "description": f"CPU Iteration: {args.num_workers} workers",
            "loader_params": {"num_workers": args.num_workers},
        },
    ]
    for config in cpu_configs:
        loader = iter(
            DataLoader(
                dataset=base_dataset,
                batch_size=args.batch_size,
                collate_fn=ex1.tokenize_collate,
                **config["loader_params"],
            ),
        )
        run_benchmark(
            description=config["description"],
            loader=loader,
            action=None,  # Time next(loader)
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
        )

    # --- 2. Benchmark GPU transfer speed ---
    if device == "cuda":
        logger.info(
            "\n--- Benchmarking GPU transfer speed (with/without pinned memory) ---",
        )
        gpu_configs = [
            {
                "description": "GPU Transfer (no pinned memory)",
                "loader_params": {"num_workers": args.num_workers, "pin_memory": False},
            },
            {
                "description": "GPU Transfer (with pinned memory)",
                "loader_params": {"num_workers": args.num_workers, "pin_memory": True},
            },
        ]
        for config in gpu_configs:
            loader = iter(
                DataLoader(
                    dataset=base_dataset,
                    batch_size=args.batch_size,
                    collate_fn=ex1.tokenize_collate,
                    **config["loader_params"],
                ),
            )
            run_benchmark(
                description=config["description"],
                loader=loader,
                action=lambda batch: batch.to(device, non_blocking=True),
                use_cuda_sync=True,
                num_steps=args.num_steps,
                warmup_steps=args.warmup_steps,
            )
