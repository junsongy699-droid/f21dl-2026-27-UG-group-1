"""Block 5: generate comparison figures (matplotlib Agg backend, runs without GUI)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = "results"
FIGURES = "figures"


def _load_regression():
    base = pd.read_csv(os.path.join(RESULTS, "regression_baselines.csv"))[["Model", "R2"]]
    mlp = pd.read_csv(os.path.join(RESULTS, "mlp_final.csv"))
    r = mlp[mlp["Task"] == "Regression"].iloc[0]
    mlp_df = pd.DataFrame({"Model": ["MLP (tuned)"], "R2": [r["Test R2"]]})
    return pd.concat([base, mlp_df], ignore_index=True).sort_values("R2")


def _load_classification():
    base = pd.read_csv(os.path.join(RESULTS, "classification_baselines.csv"))[["Model", "F1"]]
    mlp = pd.read_csv(os.path.join(RESULTS, "mlp_final.csv"))
    r = mlp[mlp["Task"] == "Classification"].iloc[0]
    mlp_df = pd.DataFrame({"Model": ["MLP (tuned)"], "F1": [r["Test F1"]]})
    return pd.concat([base, mlp_df], ignore_index=True).sort_values("F1")


def _bar(df, value_col, title, out_path, xlabel):
    models = df["Model"].tolist()
    values = df[value_col].tolist()
    colors = [
        "#d62728" if m == "MLP (tuned)" else
        ("#9e9e9e" if "Baseline" in m else "#1f77b4")
        for m in models
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(models, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(v + (max(values) * 0.01), i, f"{v:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    reg = _load_regression()
    _bar(reg, "R2", "Regression: R2 by model (higher is better)",
         os.path.join(FIGURES, "regression_r2.png"), "R2")
    clf = _load_classification()
    _bar(clf, "F1", "Classification: F1 by model (higher is better)",
         os.path.join(FIGURES, "classification_f1.png"), "F1")
    print("Generated: figures/regression_r2.png, figures/classification_f1.png")


if __name__ == "__main__":
    main()
