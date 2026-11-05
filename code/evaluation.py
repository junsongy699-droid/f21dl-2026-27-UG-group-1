"""Block 2: Experiment framework -- feature construction, data splitting, scaling, metrics."""
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30

# 13 numeric features: followers + duration_ms + 11 audio features.
NUMERIC_FEATURES = [
    "followers", "duration_ms",
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]


def build_features(df):
    """Build modelling inputs from the cleaned DataFrame.

    Returns X (n,13), y_reg (track_popularity), y_clf (explicit 0/1).
    """
    X = df[NUMERIC_FEATURES].copy()
    y_reg = df["track_popularity"].copy()
    y_clf = df["explicit"].astype(int).copy()
    return X, y_reg, y_clf


def split_data(X, y, stratify=None):
    """70/30 split with a fixed seed. Pass stratify=y for classification."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def fit_scaler(X_train):
    """StandardScaler fit on the training set only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def _self_check():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_raw, clean

    df = clean(load_raw())
    X, y_reg, y_clf = build_features(df)

    print("=" * 60)
    print("Block 2 self-check (against experimental-setup requirements)")
    print("=" * 60)

    print(f"1) X shape: {X.shape}   (expected (1000, 13))")
    print(f"   regression target range: {y_reg.min()} ~ {y_reg.max()}   (expected 0 ~ 92)")
    print(f"   classification target: {dict(y_clf.value_counts())}   (expected 0=778, 1=222)")

    Xtr, Xte, ytr, yte = split_data(X, y_reg)
    print(f"\n2) regression split: train={Xtr.shape[0]} / test={Xte.shape[0]}   (expected 700 / 300)")

    Xtr_c, Xte_c, ytr_c, yte_c = split_data(X, y_clf, stratify=y_clf)
    train_pos = ytr_c.mean()
    test_pos = yte_c.mean()
    print("\n3) stratified classification split (stratify=y):")
    print(f"   train positive ratio = {train_pos:.4f}   (expected ~0.222)")
    print(f"   test  positive ratio = {test_pos:.4f}   (expected ~0.222)")
    print(f"   consistent: {'yes' if abs(train_pos - test_pos) < 0.01 else 'no'}")

    scaler = fit_scaler(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    train_mean = np.abs(Xtr_s.mean(axis=0)).mean()
    test_mean = np.abs(Xte_s.mean(axis=0)).mean()
    train_std = Xtr_s.std(axis=0).mean()
    print("\n4) StandardScaler fit on training set only:")
    print(f"   train |mean| after transform = {train_mean:.6f}   (expected ~0)")
    print(f"   train std  after transform = {train_std:.4f}   (expected ~1)")
    print(f"   test  |mean| after transform = {test_mean:.4f}   (expected > 0)")
    print(f"   conclusion: {'PASS: no leakage' if train_mean < 1e-6 else 'FAIL: leakage detected'}")

    print("\n5) metric-function sanity checks:")
    r = regression_metrics(yte.to_numpy(), yte.to_numpy())
    print(f"   perfect regression -> RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, R2={r['R2']:.4f}   (expected 0/0/1)")
    r2 = regression_metrics(yte.to_numpy(), np.full_like(yte.to_numpy(), yte.mean()))
    print(f"   predict constant -> R2={r2['R2']:.4f}   (expected ~0)")
    c = classification_metrics(yte_c.to_numpy(), np.zeros_like(yte_c.to_numpy()))
    print(f"   classify all 0 -> Acc={c['Accuracy']:.4f}, F1={c['F1']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    _self_check()
