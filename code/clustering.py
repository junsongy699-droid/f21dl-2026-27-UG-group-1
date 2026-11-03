"""
F20DL Spotify Project - R2 K-Means Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "D1_clean.csv"
FIG_DIR = BASE_DIR / "output" / "figures"
TABLE_DIR = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    # 11 audio features only
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    X = df[features]

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'R2_elbow.png', dpi=300)
    plt.close()

    # Silhouette
    scores = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(7, 4))
    plt.plot(range(2, 11), scores, marker='s')
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'R2_silhouette.png', dpi=300)
    plt.close()

    # Save metrics
    pd.DataFrame({
        'k': range(2, 11),
        'silhouette': scores
    }).to_csv(TABLE_DIR / 'R2_clustering_metrics.csv', index=False)

    print("Done. Elbow + silhouette saved.")

if __name__ == '__main__':
    main()