from __future__ import annotations

import os
from loguru import logger


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(
        os.path.join(log_dir, "app.log"),
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
