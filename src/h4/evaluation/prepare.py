"""Prepare paired Ridge inputs without fitting or writing inputs."""

import argparse
from pathlib import Path
from ..evaluation.ridge import prepare


def prepare_for_model(root=Path.cwd()):
    """Prepare the exact paired Ridge sample without writing input files."""
    return prepare(Path(root).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", required=True)
    parser.parse_args()
    prepared, _ = prepare_for_model()
    print(prepared["checks"])
    print("Fit: python -m src.h4.evaluation.ridge")


if __name__ == "__main__":
    main()
