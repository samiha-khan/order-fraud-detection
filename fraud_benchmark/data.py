"""Loads PaySim (Lopez-Rojas et al., 2016) from Hugging Face's public mirror,
caching it locally. No account or API key needed.

PaySim is a synthetic mobile-money transaction simulator, not real data, but
it is the standard substitute the fraud-detection research community uses
specifically because real per-account transaction histories are essentially
never released publicly (privacy). See docs/fraud-benchmark.md for why this
project uses it and what its limits are.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "paysim"
SHARD_URLS = [
    "https://huggingface.co/datasets/theman10/paysim/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "https://huggingface.co/datasets/theman10/paysim/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
]


def load_paysim() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shards = []
    for i, url in enumerate(SHARD_URLS):
        path = CACHE_DIR / f"{i:04d}.parquet"
        if not path.exists():
            print(f"downloading PaySim shard {i + 1}/{len(SHARD_URLS)}...")
            urllib.request.urlretrieve(url, path)
        shards.append(pd.read_parquet(path))
    return pd.concat(shards, ignore_index=True)
