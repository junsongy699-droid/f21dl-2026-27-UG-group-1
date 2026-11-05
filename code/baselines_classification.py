"""Block 4: R3 classification baselines -- 5 classical models, predicting explicit."""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import clean, load_raw
from evaluation import build_features, classification_metrics, fit_scaler, split_data

MODELS = {
    "Logistic Regression (C=1.0)": LogisticRegression(max_iter=1000),
    "Decision Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB(),
}


def run():
    df = clean(load_raw())
    X, _, y = build_features(df)
    Xtr, Xte, ytr, yte = split_data(X, y, stratify=y)

    scaler = fit_scaler(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    rows = []
    rows.append({
        "Model": "Majority Baseline (all non-explicit)",
        **classification_metrics(yte.to_numpy(), [0] * len(yte)),
    })

    for name, model in MODELS.items():
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        rows.append({"Model": name, **classification_metrics(yte.to_numpy(), pred)})

    results = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1"]]
    results = results.sort_values("F1", ascending=False).reset_index(drop=True)
    return results


if __name__ == "__main__":
    results = run()
    print("=" * 64)
    print("R3 classification baselines (predict explicit, sorted by F1 descending)")
    print("=" * 64)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/classification_baselines.csv", index=False)
    print("\nSaved to results/classification_baselines.csv")
