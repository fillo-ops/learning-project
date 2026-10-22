"""Walk-forward feature ablation study: python -m scripts.ablation"""

from __future__ import annotations

import logging

from src.pipeline import load_or_create_dataset, run_ablation_study

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    print(run_ablation_study(load_or_create_dataset()).to_string(index=False))
