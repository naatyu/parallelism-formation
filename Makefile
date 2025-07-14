.PHONY: setup

# Set python path
export PYTHONPATH := $(shell pwd)

# Variables
PYTHON := python3
VENV_DIR := .venv
DATA_DIR := data
DATASET_DIR := $(DATA_DIR)/sample/10BT
PRETOKENIZE_DIR := $(DATA_DIR)/pretokenized_dataset

# Targets
all: env

env: $(VENV_DIR)
	@echo "Syncing env..."
	uv sync
	@echo "Syncing done !"

setup: download_data pretokenize_data
	@echo "Setup complete !"

download_data: $(VENV_DIR)
	@echo "Fetching dataset..."
	$(VENV_DIR)/bin/python src/0-setup/data_download.py --save-path $(DATA_DIR)
	@echo "Fetching dataset done !"

pretokenize_data: $(VENV_DIR)
	@echo "Pretokenizing dataset..."
	$(VENV_DIR)/bin/python src/0-setup/pre_tokenize.py --data-path $(DATASET_DIR) --save-path $(PRETOKENIZE_DIR)
	@echo "Pretokenizing dataset done !"