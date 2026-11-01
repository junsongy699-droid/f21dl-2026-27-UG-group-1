# f21dl-2026-27-UG-group-1

Spotify Song Popularity Prediction Project

## Project: Predicting Song Popularity from Audio Features & Metadata

### Course

F20DL Data Mining and Machine Learning (2026/2027)
Heriot-Watt University Malaysia

### Project Objective

This project investigates whether song popularity can be predicted using audio features and metadata extracted from Spotify. We apply data preprocessing, exploratory data analysis (EDA), clustering, baseline machine learning models and neural networks to a single Spotify dataset (~1,000 tracks), fulfilling requirements R1–R4 of the coursework.

### Dataset

| D1 | `spotifydataset.csv` : ~1,000 tracks, 23 attributes including 11 audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo), metadata (artist, genres, followers, explicit, duration_ms, release_date) and target variable `track_popularity` (range 0–92) 

The dataset is stored in the `data/` folder.

### Repository Structure

├── code/
│   ├── data_explore.py          # EDA (correlation analysis, leakage detection, distribution plots)
│   ├── data_preprocess.py       # Preprocessing pipeline (dedup, leakage removal, imputation, encoding)
│   ├── clustering.py            # k-means clustering with elbow method & silhouette score
│   ├── baseline_models.py       # Baseline regression models (Linear Regression, Ridge, Random Forest, kNN)
│   └── mlp.py                   # Multi-Layer Perceptron neural network
├── data/
│   ├── spotifydataset.csv       # Raw dataset
│   └── processed/
│       └── D1_clean.csv         # Cleaned dataset after preprocessing
├── output/
│   ├── figures/                 # EDA plots, clustering visualisations, model evaluation charts
│   └── tables/                  # Preprocessing report, evaluation metrics
└── README.md

### Requirements Covered

- **R1** Project topic and objectives defined
- **R2** Data preprocessing, cleaning, EDA, visualisation and k-means clustering
- **R3** Baseline models: Linear Regression, Ridge, Random Forest, kNN (regression task predicting `track_popularity`)
- **R4** Multi-Layer Perceptron (MLP) neural network for regression

### Generative AI Statement

In accordance with the F20DL GenAI policy:

1. Generative AI was used to **assist with code generation** for data loading, preprocessing and model boilerplate. All AI-generated code has been reviewed, tested and modified by group members. Full prompts are documented in code comments.
2. Generative AI was used **only for grammar and language proofreading** on the report. The analysis, discussion, conclusions and all main text are original work of the group.
3. No GenAI-generated content is submitted as assessment text.

### Team Contribution

| Member 1 | Yang Junsong (Leader) | Data & Exploration | Project planning, data preprocessing, EDA, clustering (R2) |
| Member 2 | He Zexin | Modelling & Evaluation | Baseline models, MLP implementation, evaluation metrics (R3, R4) |
| Member 3 | Huang Guanyin | Report & Repository | Report writing, repository maintenance, documentation |

### Data Preprocessing Pipeline (Member 1)

The preprocessing script (`code/data_preprocess.py`) performs:

- **Deduplication**: removes exact duplicate rows to prevent train-test leakage
- **Target handling**: drops rows with missing `track_popularity` (never imputed)
- **Leakage removal**: excludes columns with high correlation to target:
  - `artist_popularity` (r ≈ 1.0 with `track_popularity`, removed to prevent data leakage)
  - `followers` (moderate correlation, retained but documented as potential weak leakage)
- **Imputation**: numeric → median, categorical → 'unknown' (163 missing `genres` handled)
- **Encoding**: label-encoded `genres` column as reference for MLP; One-Hot Encoding recommended for tree-based baselines
- **Reproducibility**: fixed random seed 42, pathlib auto-detection

### Outputs

- `data/processed/D1_clean.csv` — cleaned dataset ready for modelling
- `output/tables/preprocessing_report.md` — preprocessing summary
- `output/figures/` — EDA plots, correlation heatmaps, clustering visualisations

### How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run scripts from the `code/` folder in order:
   `data_explore.py` → `data_preprocess.py` → `clustering.py` → `baseline_models.py` → `mlp.py`

### Known Limitations

- `track_popularity` range 0–92; prediction is a regression task, not classification
- `explicit` class imbalance (~222 explicit vs ~778 non-explicit) may affect any auxiliary classification experiments
- Label encoding on `genres` implies ordinality; One-Hot Encoding is recommended for MLP input
- Dataset size (~1,000 tracks) limits model generalisation; cross-validation used to mitigate