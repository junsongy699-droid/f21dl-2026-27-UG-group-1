"""Block 3: R3 regression baselines -- 5 classical models evaluated uniformly."""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import clean, load_raw
from evaluation import build_features, fit_scaler, regression_metrics, split_data

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Decision Tree (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest (n=100)": RandomForestRegressor(n_estimators=100, random_state=42),
    "kNN (k=5)": KNeighborsRegressor(n_neighbors=5),
}


def run():
    df = clean(load_raw())
    X, y, _ = build_features(df)
    Xtr, Xte, ytr, yte = split_data(X, y)

    scaler = fit_scaler(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    rows = []
    mean_pred = ytr.mean()
    rows.append({
        "Model": "Mean Baseline (predict train mean)",
        **regression_metrics(yte.to_numpy(), [mean_pred] * len(yte)),
    })

    for name, model in MODELS.items():
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        rows.append({"Model": name, **regression_metrics(yte.to_numpy(), pred)})

    results = pd.DataFrame(rows)[["Model", "RMSE", "MAE", "R2"]]
    results = results.sort_values("RMSE").reset_index(drop=True)
    return results


if __name__ == "__main__":
    results = run()
    print("=" * 60)
    print("R3 regression baselines (sorted by RMSE ascending)")
    print("=" * 60)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/regression_baselines.csv", index=False)
    print("\nSaved to results/regression_baselines.csv")
