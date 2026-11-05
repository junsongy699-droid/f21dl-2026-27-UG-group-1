"""One-command reproduction of the modelling pipeline (R3 + R4).

Usage:
    python code/run_all.py

Runs: data verification -> regression baselines -> classification baselines ->
MLP tuning -> figures.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "data_loader.py",
    "baselines_regression.py",
    "baselines_classification.py",
    "mlp_model.py",
    "figures.py",
]


def main():
    for name in SCRIPTS:
        print("\n" + "#" * 60)
        print(f"# Running {name}")
        print("#" * 60)
        subprocess.run([sys.executable, os.path.join(HERE, name)], check=True, cwd=HERE)

    print("\n" + "#" * 60)
    print("# All done")
    print("#" * 60)


if __name__ == "__main__":
    main()
