"""
F20DL Spotify Project - Data Preprocessing Script (R2 Deliverable)
AI Usage Declaration:
This script was iteratively refined with GenAI assistance based on code review.
Full prompt used for generation:
"Write a robust data preprocessing script for F20DL Spotify popularity prediction project.
Requirements:
- Two datasets D1 (global trends) and D2 (tracks metadata)
- Remove duplicates
- Drop rows with missing target variable (popularity), never fill target
- Drop known leakage features with exact match, avoid false positive deletion
- Fill numeric missing values with median, categorical with 'unknown'
- Output genre label-encoded column as reference
- Generate full markdown preprocessing report for R2 write-up
- Generate popularity range record text file
- Auto-detect CSV filenames with fuzzy matching
- Add friendly error messages for missing columns
- Use pathlib for all paths, fixed random seed 42
- Include full AI usage declaration per course policy"
All logic, rules and output paths were manually reviewed and validated by group members.
"""



import pandas as pd
import numpy as np
from pathlib import Path
import warnings
# Suppress unnecessary pandas warnings for cleaner output
warnings.filterwarnings("ignore")

# ========== Global Configuration ==========
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LEAKAGE_EXACT_D1 = ["artist_popularity", "streams", "pos", "stream_change", "7day", "viral_score"]
LEAKAGE_EXACT_D2 = ["artist_popularity"]

# ========== Automatic Path Localization ==========
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
# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def resolve_csv(keyword_hint):
    """
    Fuzzy match CSV files in the data directory. 
    Provides detailed error messages if files are missing or ambiguous.
    """
    matches = list(DATA_DIR.glob(f"*{keyword_hint}*.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple matches for '{keyword_hint}' in data/: {[m.name for m in matches]}\n"
            "Please specify exact filename."
        )
    all_files = [f.name for f in DATA_DIR.glob("*.csv")] if DATA_DIR.exists() else []
    raise FileNotFoundError(
        f"No CSV matching '{keyword_hint}' found in {DATA_DIR}\n"
        f"Available CSV files: {all_files}"
    )

def find_popularity_column(df):
    """Safely locate the target column 'popularity', providing fallback options."""
    # Exact match (case-insensitive) preferred
    candidates = [c for c in df.columns if c.lower().strip() == "popularity"]
    if len(candidates) == 1:
        return candidates[0]
    
    # Fallback: substring match
    candidates = [c for c in df.columns if "popularity" in c.lower()]
    if len(candidates) >= 1:
        return candidates[0]
        
    raise ValueError(
        "Cannot find target column 'popularity' in dataset.\n"
        f"Available columns: {list(df.columns)}"
    )

def preprocess_dataset(df, leakage_cols, dataset_name):
    """Core data cleaning pipeline for a single dataset."""  
    report_lines = [f"## {dataset_name}", ""]
    
    initial_rows, initial_cols = df.shape
    report_lines.append(f"- **Initial shape**: {initial_rows} rows × {initial_cols} columns")

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    dup_removed = initial_rows - len(df)
    report_lines.append(f"- **Duplicates removed**: {dup_removed} rows")

    #  Drop rows with missing target variable (Never impute the target)
    pop_col = find_popularity_column(df)
    before_target = len(df)
    df = df.dropna(subset=[pop_col]).reset_index(drop=True)
    target_dropped = before_target - len(df)
    report_lines.append(f"- **Missing target dropped**: {target_dropped} rows (column: `{pop_col}`)")

    # Remove leakage features (exact match)
    dropped_leakage = []
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped_leakage.append(col)
    report_lines.append(
        f"- **Leakage features removed**: {', '.join(dropped_leakage) if dropped_leakage else 'None'}"
    )

    # Next steps: Missing value imputation and encoding will be added in the next commit ---
    
    return df, report_lines

if __name__ == "__main__":
    path_d1 = resolve_csv("global_trends")
    df1 = pd.read_csv(path_d1)
    df1_clean, report = preprocess_dataset(df1, LEAKAGE_EXACT_D1, "D1")
    print("Cleaning report:", report)
    print(df1_clean.head())