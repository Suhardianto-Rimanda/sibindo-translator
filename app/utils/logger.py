import logging
import sys


def setup_logger(name: str = "sibindo", level: int = logging.INFO) -> logging.Logger:
    """Centralized logger config. Use across services."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = setup_logger()
