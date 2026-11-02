"""Shared paths and loaders for the case study scripts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ytmod.clean import clean_text, hash_author  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
for sub in ("results", "tables", "figures"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)


def load_scam_comments() -> pd.DataFrame:
    df = pd.read_csv(RAW / "scam-youtube-dataset-comments.csv")
    df["raw_content"] = df["content"]
    df["text"] = df["content"].map(clean_text)
    df["author_id"] = df["author"].map(hash_author)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.drop(columns=["content", "author"]).reset_index(names="comment_row")


def load_videos() -> pd.DataFrame:
    df = pd.read_csv(RAW / "scam-youtube-video-ids.csv")
    df["title"] = df["title"].map(clean_text)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


def load_toxic() -> pd.DataFrame:
    df = pd.read_csv(RAW / "youtoxic_english_1000.csv")
    df["text"] = df["Text"].map(clean_text)
    return df


def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv(RAW / "YoutubeCommentsDataSet.csv")
    df = df.rename(columns={"Comment": "text_raw", "Sentiment": "sentiment"})
    df["text"] = df["text_raw"].fillna("").map(clean_text)
    return df


def to_py(obj):
    """Make numpy and pandas values JSON-friendly."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_json(name: str, data: dict) -> None:
    (OUT / "results" / f"{name}.json").write_text(json.dumps(to_py(data), indent=2), encoding="utf-8")


def load_json(name: str) -> dict:
    return json.loads((OUT / "results" / f"{name}.json").read_text(encoding="utf-8"))
