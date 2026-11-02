"""Executa apenas relatórios de modelos existentes, sem repetir treinamento."""

import argparse
import os
import time
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["xgboost", "random_forest", "test", "all"], default="all"
    )
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
        # Evita truncar o notebook em falhas transitórias de escrita/sincronização.
        temporary = path.with_name(f".{path.stem}.{uuid4().hex[:8]}.executed.ipynb")
        nbformat.write(notebook, temporary)
        nbformat.validate(nbformat.read(temporary, as_version=4))
        for attempt in range(3):
            try:
                temporary.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.5)
        print(f"Notebook salvo: {path.name}", flush=True)
        if model == "test":
            destination = root / "reports/notebook_snapshots/final_test"
            destination.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, destination / path.name)


if __name__ == "__main__":
    main()
