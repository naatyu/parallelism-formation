# Data Loading

The goal of is section is to compare the different datasets type and loader parameters.

# Exercise 1

This exercise implement a `MapDataset`, the classic dataset class of PyTorch.
Complete `__len__` and `__getitem__` methods of this `MapDataset`. Use `make data_loading_ex1` to check if your code is correct.

# Exercise 2

This exercise implement an `IterableDataset`. The goal is to load the data from a memory mapped file as seen in the lecture.


# Benchmark

When done with the previous exercises run the benchmark and see the difference between various parameters with:
```bash
make data_loading_benchmark
```