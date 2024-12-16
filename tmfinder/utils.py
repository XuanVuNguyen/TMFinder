import logging
from typing import Optional


def init_logger(name, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        # file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
