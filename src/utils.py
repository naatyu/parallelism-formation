"""General functions to use across the project such as setting up logger."""

import logging
from pathlib import Path


def get_logger(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Configure and return a logger instance for your ML pipeline.

    Args:
        log_file (str): Path to the log file.
        level (int): Minimum logging level for the logger (e.g., logging.INFO, logging.DEBUG).
        enable_file_logging (bool): If True, a file handler will be added.

    Returns:
        logging.Logger: The configured logger instance.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            # Ensure the log directory exists before creating the file
            if not log_file.exists():
                log_file.parent.mkdir(exist_ok=True, parents=True)

            file_handler = logging.FileHandler(str(log_file), mode="a")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


logger = get_logger()
