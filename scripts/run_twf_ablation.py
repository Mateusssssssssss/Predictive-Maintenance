"""Ablação de features por CV no FIT_SEARCH (sem utilizar VALIDATION/TEST)."""

from pathlib import Path

from predictive_maintenance.twf_experiment import run_twf_ablation


if __name__ == "__main__":
    run_twf_ablation(Path(__file__).resolve().parents[1])
