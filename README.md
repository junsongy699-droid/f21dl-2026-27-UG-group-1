# f21dl-2026-27-UG-group-1
Spotify Song Popularity Prediction Project
## Project: Predicting Song Popularity from Audio Features & Metadata

### Course
F20DL Data Mining and Machine Learning (2026/2027)  
Heriot-Watt University Malaysia

### Project Objective
This project investigates whether song popularity can be predicted using audio features and metadata. We apply data preprocessing, exploratory data analysis (EDA), clustering, baseline machine learning models and neural networks to two Spotify datasets, fulfilling requirements R1–R4 of the coursework.

### Datasets
| D1 | `spotifydataset.csv` | ~1,000 tracks with 23 attributes including 11 audio features, popularity range 0–92 |
| D2 | `spotify_tracks.csv` | ~6,300 tracks with metadata fields, genre labels and popularity range 0–90 |

Both datasets are stored in the `data/` folder.

### Repository Structure

├── code/    
|    ├── data_explore.py          # EDA (Step 1-5, leakage detection correlation)
│    ├── data_preprocess.py
|    # Python scripts for preprocessing, EDA, modelling   and neural networks
├── data/          # Raw and cleaned datasets
├── output/tables/        # Generated tables, figures and evaluation 
     └── preprocessing_report.md
└── README.md




### Requirements Covered
- **R1** Project topic and objectives defined
- **R2** Data preprocessing, cleaning, EDA, visualisation and k-means clustering
- **R3** Baseline models: linear regression, decision tree, kNN, naive Bayes, random forest
- **R4** Multi-Layer Perceptron (MLP) neural network

### Generative AI Statement
In accordance with the F20DL GenAI policy:
1. Generative AI was used to **assist with code generation** for data loading, preprocessing and model boilerplate. All AI-generated code has been reviewed, tested and modified by group members. Full prompts are documented in code comments.
2. Generative AI was used **only for grammar and language proofreading** on the report. The analysis, discussion, conclusions and all main text are original work of the group.
3. No GenAI-generated content is submitted as assessment text.

### Team Contribution
| Member 1: Yang Junsong | Leader / Data & Exploration | Project planning, data preprocessing, EDA, clustering (R2) |
| Member 2: He Zexin | Modelling & Evaluation | Baseline models, MLP implementation, evaluation metrics (R3, R4) |
| Member 3: Huang Guanyin | Report & Repository | Report writing, repository maintenance, documentation (D2) |

### Data Preprocessing Pipeline (Member 1)
The preprocessing script (`code/data_preprocess.py`) performs:
- **Deduplication**: removes exact duplicate rows to prevent train-test leakage
- **Target handling**: drops rows with missing `popularity` (never imputed)
- **Leakage removal**: exact-match exclusion of 6 columns in D1 
  (`artist_popularity`, `streams`, `pos`, `stream_change`, `7day`, `viral_score`)
  and 1 column in D2 (`artist_popularity`)
- **Imputation**: numeric → median, categorical → 'unknown' (163 missing genres handled)
- **Encoding**: label-encoded `genre` column as reference for MLP
- **Reproducibility**: fixed random seed 42, pathlib auto-detection, fuzzy CSV matching

### Data Preprocessing Pipeline (Member 2)

### Data Preprocessing Pipeline (Member 3)

### Outputs:`data/processed/D1_clean.csv`, `D2_clean.csv`, 
`output/tables/R2_preprocessing_report.md`

### How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run scripts from the `code/` folder in order: `data_explore.py` → `preprocessing.py` → `modelling.py` → `mlp.py`

### Known Limitations
- D1 and D2 popularity ranges differ (0–92 vs 0–90) due to API collection timing
- Label encoding on genre implies ordinality; One-Hot recommended for MLP
- D1 explicit class imbalance (222:778) may affect classification baselines