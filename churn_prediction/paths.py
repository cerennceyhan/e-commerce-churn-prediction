"""Central place for every file-system location used by the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

SAMPLE_DATASET = RAW_DIR / "sample_dataset.csv"
SCRAPED_DATASET = RAW_DIR / "dataset-first.csv"
PRODUCT_LINKS = CONFIG_DIR / "product_links.txt"

BASE_METRICS = PROCESSED_DIR / "base_metrics.csv"
LLM_RESULTS = PROCESSED_DIR / "llm_results.csv"
LLM_EXTRACTION = PROCESSED_DIR / "llm_extraction.csv"
