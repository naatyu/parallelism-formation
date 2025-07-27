.PHONY: setup

# Set env
export PYTHONPATH := $(shell pwd)
export TOKENIZERS_PARALLELISM := false

# Variables
PYTHON := python3
VENV_DIR := .venv
DATA_DIR := data
DATASET_DIR := $(DATA_DIR)/sample/10BT
PRETOKENIZE_DIR := $(DATA_DIR)/pretokenized_dataset

all: env

# --- Setup --- #
env: $(VENV_DIR)
	@echo "Syncing env..."
	uv sync
	@echo "Syncing done !"

setup: download_data pretokenize_data
	@echo "Setup complete !"

download_data: $(VENV_DIR)
	@echo "Fetching dataset..."
	$(VENV_DIR)/bin/python src/setup_0/data_download.py --save-path $(DATA_DIR)
	@echo "Fetching dataset done !"

pretokenize_data: $(VENV_DIR)
	@echo "Pretokenizing dataset..."
	$(VENV_DIR)/bin/python src/setup_0/pre_tokenize.py --data-path $(DATASET_DIR) --save-path $(PRETOKENIZE_DIR)
	@echo "Pretokenizing dataset done !"

# --- Exercises --- #
data_loading_ex1: $(VENV_DIR)
	@echo "Runing exercise 1..."
	$(VENV_DIR)/bin/python src/data_loading_1/ex1.py

data_loading_benchmark: $(VENV_DIR)
	@echo "Running data loading benchmark..."
	$(VENV_DIR)/bin/python src/data_loading_1/benchmark.py