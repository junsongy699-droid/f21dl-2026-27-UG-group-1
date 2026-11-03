"""
F20DL Spotify Project - Data Exploration Script (Single Dataset)
AI Usage Declaration:
Generated with GenAI assistance. 
Prompt: Write a pandas EDA script for the spotifydataset.csv, output missing stats, describe numeric cols, calculate correlation with track_popularity, and save to output/tables.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    path = DATA_DIR / "spotifydataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nPlease ensure spotifydataset.csv is in data/")

    print(f"Loading dataset: {path.name}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Missing value statistics
    miss = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_rate_pct": round(df.isnull().sum() / len(df) * 100, 2)
    }).sort_values("missing_count", ascending=False)
    print("\n--- Missing Value Statistics ---")
    print(miss[miss["missing_count"] > 0])
    miss.to_csv(TABLE_DIR / "D1_missing_stats.csv", encoding="utf-8-sig")

    # 2. Numeric Description
    desc = df.describe().T
    desc.to_csv(TABLE_DIR / "D1_numeric_stats.csv", encoding="utf-8-sig")
    print("\n--- Numeric Description saved to output/tables/ ---")

    # 3. Correlation with Target (Crucial for R2 leakage check)
    print("\n--- Correlation with track_popularity ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if "track_popularity" in num_cols:
        corr = df[num_cols].corr()["track_popularity"].sort_values(ascending=False)
        print(corr)
        corr.to_csv(TABLE_DIR / "D1_correlation_target.csv", encoding="utf-8-sig")

    print("\nEDA finished. All results saved to output/tables/")

if __name__ == "__main__":
    main()