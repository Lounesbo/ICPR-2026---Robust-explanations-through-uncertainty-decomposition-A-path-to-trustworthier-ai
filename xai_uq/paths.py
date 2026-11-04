"""Project paths, resolved from the package location.

This lets the scripts run from any working directory.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUT_DIR = ROOT_DIR / "output"
