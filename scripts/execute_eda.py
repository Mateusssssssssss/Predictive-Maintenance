"""Executa EDA e salva notebook com saídas; não ajusta modelos."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Mantém os caches e arquivos de execução dentro do projeto.
    for variable, relative in {
        "JUPYTER_RUNTIME_DIR": ".jupyter/runtime",
        "JUPYTER_CONFIG_DIR": ".jupyter/config",
        "JUPYTER_DATA_DIR": ".jupyter/data",
        "IPYTHONDIR": ".ipython",
        "MPLCONFIGDIR": ".cache/matplotlib",
    }.items():
        os.environ.setdefault(variable, str(root / relative))
    path = root / "notebooks/01_eda.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Notebook executado: {path}")


if __name__ == "__main__":
    main()
