"""
F20DL Spotify Project - Data Preprocessing Script (Single Dataset)
AI Usage Declaration:
This script was iteratively refined with GenAI assistance. 
All logic, parameters, and output paths manually reviewed.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Drop columns that cause data leakage or model errors
LEAKAGE_EXACT = [
    "Unnamed: 0",         # CSV index column
    "artist_popularity",  # Severe data leakage (highly correlated with target)
    "artist_url",         # Useless URL text
    "artist_name",        # Text column
    "track_name",         # Text column
    "album_name"          # Text column
]

def find_project_root(start=None):
    current = (start or Path(__file__).resolve()).parent
    for p in [current, *current.parents]:
        if (p / "data").is_dir():
            return p
    return current.parent

BASE_DIR = find_project_root()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "output" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def find_popularity_column(df):
    """Safely locate the target column, hardcoded to track_popularity."""
    if "track_popularity" in df.columns:
        return "track_popularity"
    # Fallback if naming changed
    candidates = [c for c in df.columns if "popularity" in c.lower() and "artist" not in c.lower()]
    if len(candidates) > 0:
        return candidates[0]
    raise ValueError("Cannot find target column 'track_popularity'.")

def preprocess_dataset(df):
    """Core data cleaning pipeline."""
    report_lines = []
    initial_rows, initial_cols = df.shape
    report_lines.append(f"- **Initial shape**: {initial_rows} rows x {initial_cols} columns")

    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    dup_removed = initial_rows - len(df)
    report_lines.append(f"- **Duplicates removed**: {dup_removed} rows")

    # 2. Drop rows with missing target
    pop_col = find_popularity_column(df)
    before_target = len(df)
    df = df.dropna(subset=[pop_col]).reset_index(drop=True)
    target_dropped = before_target - len(df)
    report_lines.append(f"- **Missing target dropped**: {target_dropped} rows (column: `{pop_col}`)")

    # 3. Remove leakage / useless text columns
    dropped_cols = []
    for col in LEAKAGE_EXACT:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped_cols.append(col)
    report_lines.append(f"- **Leakage/Text features removed**: {', '.join(dropped_cols)}")

    # 4. Feature Engineering: Release Date -> Release Year
    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        # Fill NaNs resulting from date parsing errors
        df["release_year"] = df["release_year"].fillna(df["release_year"].median())
        df = df.drop(columns=["release_date"])
        report_lines.append("- **Feature Engineering**: Extracted `release_year` from `release_date`")

    # 5. Handle Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_missing_total = int(df[num_cols].isnull().sum().sum())
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    cat_missing_total = int(df[cat_cols].isnull().sum().sum())
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("unknown")

    report_lines.append(f"- **Missing values filled**: {num_missing_total} numeric (median), {cat_missing_total} categorical ('unknown')")

    final_rows, final_cols = df.shape
    report_lines.append(f"- **Final shape**: {final_rows} rows x {final_cols} columns")

    return df, report_lines

def main():
    path = DATA_DIR / "spotifydataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading {path.name}...")
    df_raw = pd.read_csv(path)
    
    df_clean, report = preprocess_dataset(df_raw)
    
    # Save outputs
    out_path = OUTPUT_DIR / "D1_clean.csv"
    df_clean.to_csv(out_path, index=False)
    
    print("\n--- Cleaning Report ---")
    for line in report:
        print(line)
        
    print(f"\n[SUCCESS] Cleaned data saved to {out_path}")
    print(f"Columns in cleaned dataset: {list(df_clean.columns)}")

if __name__ == "__main__":
    main()