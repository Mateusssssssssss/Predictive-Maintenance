"""Executa Optuna, variáveis derivadas, calibração e ajuste de limiar nos dois modelos."""

import argparse
import os
from pathlib import Path

from predictive_maintenance.advanced import run_advanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "xgboost", "random_forest"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(root / ".cache/joblib"))
    (root / ".cache/joblib").mkdir(parents=True, exist_ok=True)
    for name in ["xgboost", "random_forest"] if args.model == "all" else [args.model]:
        run_advanced(root, name)
