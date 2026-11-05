"""Block 1: Data loading and verification.

Loads the real spotifydataset.csv from the repository's data/ folder, drops the
leftover index column and the leakage feature, and prints verification info.
"""
import os

import pandas as pd

# Resolve the raw dataset relative to this file: code/ -> repo root/data/spotifydataset.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spotifydataset.csv")

# Known facts (for verification).
TARGET = "track_popularity"
LEAKAGE = "artist_popularity"
INDEX_COL = "Unnamed: 0"


def load_raw() -> pd.DataFrame:
    """Read the raw CSV from the repository data folder."""
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: drop leftover index column + leakage feature."""
    df = df.copy()
    if INDEX_COL in df.columns:
        df = df.drop(columns=[INDEX_COL])
    if LEAKAGE in df.columns:
        df = df.drop(columns=[LEAKAGE])
    return df


def verify(raw: pd.DataFrame) -> None:
    print("=" * 60)
    print("Data verification (against known facts)")
    print("=" * 60)

    print(f"1) Shape: {raw.shape[0]} rows x {raw.shape[1]} cols   (expected 1000 x 23)")

    print(f"2) First column: {raw.columns[0]!r}   (expected 'Unnamed: 0')")

    if LEAKAGE in raw.columns and TARGET in raw.columns:
        corr = raw[LEAKAGE].corr(raw[TARGET])
        equal = int((raw[LEAKAGE] == raw[TARGET]).sum())
        print("3) artist_popularity vs track_popularity:")
        print(f"   - correlation r = {corr:.6f}   (expected 1.0)")
        print(f"   - equal rows = {equal}/{len(raw)}   (expected 1000/1000)")

    if "explicit" in raw.columns:
        vc = raw["explicit"].value_counts()
        print(f"4) explicit distribution: {dict(vc)}   (expected False 778 / True 222)")

    if "genres" in raw.columns:
        miss = int(raw["genres"].isna().sum())
        print(f"5) genres missing: {miss} ({miss/len(raw)*100:.1f}%)   (expected 163 / 16.3%)")

    if "release_date" in raw.columns:
        d = pd.to_datetime(raw["release_date"], errors="coerce")
        print(f"6) release_date range: {d.min().date()} ~ {d.max().date()}   (expected 1971 ~ 2024)")

    if TARGET in raw.columns:
        print(f"7) track_popularity range: {raw[TARGET].min()} ~ {raw[TARGET].max()}   (expected 0 ~ 92)")

    print("=" * 60)
    print(f"All columns ({len(raw.columns)}):")
    print(list(raw.columns))


if __name__ == "__main__":
    raw = load_raw()
    verify(raw)

    cleaned = clean(raw)
    print("\nAfter cleaning (drop Unnamed:0 + artist_popularity):")
    print(f"  shape: {cleaned.shape}")
    numeric = cleaned.select_dtypes(include=["number"]).columns.tolist()
    print(f"  numeric columns ({len(numeric)}): {numeric}")
    print(f"  non-numeric columns: {[c for c in cleaned.columns if c not in numeric]}")
