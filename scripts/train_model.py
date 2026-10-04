"""Run the 'train' step of the pipeline. Usage: python scripts/train_model.py [-v]"""

import sys

from metricmind.interfaces.cli.commands import run_command

if __name__ == "__main__":
    raise SystemExit(run_command("train", sys.argv[1:]))
