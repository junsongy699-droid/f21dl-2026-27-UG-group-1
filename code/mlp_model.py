"""Block 5: R4 neural networks -- MLPRegressor / MLPClassifier tuning + evaluation."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import clean, load_raw
from evaluation import build_features, classification_metrics, regression_metrics, split_data

PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "mlp__alpha": [0.0001, 0.01],
    "mlp__learning_rate_init": [0.001, 0.01],
}


def _tuning_table(grid: GridSearchCV) -> pd.DataFrame:
    cv = pd.DataFrame(grid.cv_results_)
    cols = [
        "param_mlp__hidden_layer_sizes",
        "param_mlp__alpha",
        "param_mlp__learning_rate_init",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    tbl = cv[cols].copy()
    tbl.columns = [
        "hidden_layer_sizes", "alpha", "learning_rate_init",
        "mean_cv_score", "std_cv_score", "rank",
    ]
    return tbl.sort_values("rank")


def tune_mlp_regression(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(max_iter=5000, random_state=42, early_stopping=True, n_iter_no_change=20)),
    ])
    grid = GridSearchCV(pipe, PARAM_GRID, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def tune_mlp_classification(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=5000, random_state=42, early_stopping=True, n_iter_no_change=20)),
    ])
    grid = GridSearchCV(pipe, PARAM_GRID, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def main():
    df = clean(load_raw())
    X, y_reg, y_clf = build_features(df)
    os.makedirs("results", exist_ok=True)

    Xtr, Xte, ytr, yte = split_data(X, y_reg)
    print("=" * 60)
    print("R4 MLP regression tuning (GridSearchCV + 5-fold)")
    print("=" * 60)
    g_reg = tune_mlp_regression(Xtr, ytr)
    print(f"Best params: {g_reg.best_params_}")
    print(f"Best 5-fold CV RMSE: {-g_reg.best_score_:.4f}")
    best_reg = g_reg.best_estimator_
    reg_test = regression_metrics(yte.to_numpy(), best_reg.predict(Xte))
    print(f"Test: RMSE={reg_test['RMSE']:.4f}, MAE={reg_test['MAE']:.4f}, R2={reg_test['R2']:.4f}")
    tbl_reg = _tuning_table(g_reg)
    tbl_reg.to_csv("results/mlp_tuning_regression.csv", index=False)

    Xtr_c, Xte_c, ytr_c, yte_c = split_data(X, y_clf, stratify=y_clf)
    print("\n" + "=" * 60)
    print("R4 MLP classification tuning (GridSearchCV + 5-fold)")
    print("=" * 60)
    g_clf = tune_mlp_classification(Xtr_c, ytr_c)
    print(f"Best params: {g_clf.best_params_}")
    print(f"Best 5-fold CV F1: {g_clf.best_score_:.4f}")
    best_clf = g_clf.best_estimator_
    clf_test = classification_metrics(yte_c.to_numpy(), best_clf.predict(Xte_c))
    print(f"Test: Acc={clf_test['Accuracy']:.4f}, Prec={clf_test['Precision']:.4f}, "
          f"Recall={clf_test['Recall']:.4f}, F1={clf_test['F1']:.4f}")
    tbl_clf = _tuning_table(g_clf)
    tbl_clf.to_csv("results/mlp_tuning_classification.csv", index=False)

    final = pd.DataFrame({
        "Task": ["Regression", "Classification"],
        "Best Params": [str(g_reg.best_params_), str(g_clf.best_params_)],
        "CV Score": [-g_reg.best_score_, g_clf.best_score_],
        "Test RMSE": [reg_test["RMSE"], np.nan],
        "Test MAE": [reg_test["MAE"], np.nan],
        "Test R2": [reg_test["R2"], np.nan],
        "Test Accuracy": [np.nan, clf_test["Accuracy"]],
        "Test Precision": [np.nan, clf_test["Precision"]],
        "Test Recall": [np.nan, clf_test["Recall"]],
        "Test F1": [np.nan, clf_test["F1"]],
    })
    final.to_csv("results/mlp_final.csv", index=False)
    print("\nSaved MLP results.")


if __name__ == "__main__":
    main()
