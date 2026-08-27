"""Logging infrastructure with console and file handlers."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Optional


def setup_logger(
    name: str = "omilos",
    log_dir: Optional[str | Path] = None,
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger with console and optional file handlers.

    Args:
        name: Name of the logger instance.
        log_dir: Directory where log files should be stored. Defaults to 'logs/'.
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_to_file: Whether to attach a rotating/timestamped file handler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.handlers:
        return logger

    # Log format: Timestamp | Level | Name:Line | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Stream Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (optional)
    if log_to_file:
        target_dir = Path(log_dir) if log_dir else Path("logs")
        target_dir.mkdir(parents=True, exist_ok=True)

        log_file = target_dir / "app.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate log entries
    logger.propagate = False

    return logger
