"""Treina cada candidato com GridSearchCV e persiste artefatos para os notebooks."""

import argparse
import os
from pathlib import Path

from predictive_maintenance.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["xgboost", "random_forest", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(root / ".cache/joblib"))
    (root / ".cache/joblib").mkdir(parents=True, exist_ok=True)
    for name in ["xgboost", "random_forest"] if args.model == "all" else [args.model]:
        run_experiment(root, name)


if __name__ == "__main__":
    main()
