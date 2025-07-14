"""The script is used to download the dataset (FineWeb-Edu 10B tokens sample)."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=Path,
        dest="save_path",
        help="Where to save the dataset.",
        required=True,
    )
    args = parser.parse_args()

    # Download to desired dir
    res = snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns="sample/10BT/*",
        local_dir=args.save_path,
    )

    logger.info(f"Successfully downloaded FineWeb-Edu 10B in {res} !")
