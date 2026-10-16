"""
F20DL Spotify Project - Data Exploration Script
AI Usage Declaration:
This script template was generated with AI assistance.
Prompt: Write a pandas data exploration script for two spotify datasets, output missing value statistics, describe numeric columns, compare common fields, save results to output/tables.
Part of the code was produced by GenAI. All logic, parameters, dataset file paths and other configurations were manually reviewed, revised and adjusted by group members.
"""

#version 1.0 
#The two datasets only share the `genre` categorical column; no common numeric features exist.

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ========== Global configuration ==========
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ========== Project paths ==========
# This script is expected to live in code/, so parent.parent is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file names (must match the actual files in data/)
PATH_D1 = DATA_DIR / "spotify_global_trends_2026 - spotify_global_trends_final.csv"
PATH_D2 = DATA_DIR / "spotify_tracks.csv"


# ========== Helper: validate file paths ==========
def validate_paths(paths):
    """Raise a clear error if any input file is missing."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"File not found: {p}\n"
                f"Files currently in data/ folder: "
                f"{[f.name for f in DATA_DIR.glob('*.csv')]}"
            )


# ========== Core: basic exploration for one dataset ==========
def explore_dataset(df, name):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Column dtypes
    print("\n--- Column dtypes ---")
    print(df.dtypes)

    # Missing value statistics
    miss = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_rate_pct": round(df.isnull().sum() / len(df) * 100, 2)
    }).sort_values("missing_count", ascending=False)
    print("\n--- Missing value statistics ---")
    print(miss)
    miss.to_csv(TABLE_DIR / f"{name}_missing_stats.csv", encoding="utf-8-sig")
    print(f"Saved: {TABLE_DIR / f'{name}_missing_stats.csv'}")

    # Numeric column descriptive statistics
    print("\n--- Numeric columns description ---")
    desc = df.describe().T
    print(desc)
    desc.to_csv(TABLE_DIR / f"{name}_numeric_stats.csv", encoding="utf-8-sig")

    return miss, desc


# ========== Core: compare common columns between the two datasets ==========
def compare_common_columns(df1, df2):
    common_cols = sorted(set(df1.columns) & set(df2.columns))
    print(f"\n{'='*60}")
    print(f"Common columns between D1 & D2: {common_cols}")

    compare_rows = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            compare_rows.append({
                "column": col,
                "D1_min": df1[col].min(),
                "D1_max": df1[col].max(),
                "D1_mean": round(df1[col].mean(), 4),
                "D2_min": df2[col].min(),
                "D2_max": df2[col].max(),
                "D2_mean": round(df2[col].mean(), 4)
            })

    compare_df = pd.DataFrame(compare_rows)
    print("\n--- Common numeric columns range comparison ---")
    print(compare_df)
    compare_df.to_csv(
        TABLE_DIR / "D1_D2_common_columns_compare.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Saved: {TABLE_DIR / 'D1_D2_common_columns_compare.csv'}")

    # Highlight the popularity range difference (needed for the report)
    if "popularity" in common_cols:
        print("\nKey difference: popularity range")
        print(f"D1 popularity: min={df1['popularity'].min()}, max={df1['popularity'].max()}")
        print(f"D2 popularity: min={df2['popularity'].min()}, max={df2['popularity'].max()}")


# ========== Main ==========
def main():
    # Step 0: quick debug info
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("PATH_D1:", PATH_D1)
    print("PATH_D2:", PATH_D2)
    print("data/ csv files:", [f.name for f in DATA_DIR.glob("*.csv")])

    # Step 1: validate input files
    validate_paths([PATH_D1, PATH_D2])

    # Step 2: load datasets
    df1 = pd.read_csv(PATH_D1)
    df2 = pd.read_csv(PATH_D2)

    # Step 3: explore each dataset
    miss1, desc1 = explore_dataset(df1, "D1")
    miss2, desc2 = explore_dataset(df2, "D2")

    # Step 4: cross-dataset comparison
    compare_common_columns(df1, df2)

    print("\nData exploration finished. All results saved to output/tables/")


if __name__ == "__main__":
    main()