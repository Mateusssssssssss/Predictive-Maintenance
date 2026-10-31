"""Executa apenas relatórios de modelos existentes, sem repetir treinamento."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["xgboost", "random_forest", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for variable, relative in {
        "JUPYTER_RUNTIME_DIR": ".jupyter/runtime",
        "JUPYTER_CONFIG_DIR": ".jupyter/config",
        "JUPYTER_DATA_DIR": ".jupyter/data",
        "IPYTHONDIR": ".ipython",
        "MPLCONFIGDIR": ".cache/matplotlib",
    }.items():
        os.environ.setdefault(variable, str(root / relative))
    models = ("xgboost", "random_forest") if args.model == "all" else (args.model,)
    for model in models:
        path = root / "notebooks" / f"metrics_{model}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        print(f"Notebook salvo: {path.name}", flush=True)


if __name__ == "__main__":
    main()
