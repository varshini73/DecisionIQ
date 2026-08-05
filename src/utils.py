"""
utils.py
Shared paths, logging, and small generic helpers used across the pipeline.
Business assumptions live in config.py, not here -- see that file's docstring
for why the split matters.
"""

import logging
from pathlib import Path

from config import CONFIG  # re-exported for convenience: `from utils import CONFIG`

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_EXTERNAL_DIR = ROOT_DIR / "data" / "external"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = ROOT_DIR / "data" / "processed" / "decisioniq.db"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXTERNAL_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
