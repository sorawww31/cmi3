import logging
import time
from logging import INFO, FileHandler, StreamHandler
from pathlib import Path


def get_logger(file_name: str, file_dir: Path | str) -> logging.Logger:
    logger = logging.getLogger(file_name)

    # 既存のハンドラがある場合は重複追加を避ける
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s : %(levelname)s - %(filename)s] %(message)s")

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = Path(file_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = FileHandler(log_file)
    file_handler.setLevel(INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger