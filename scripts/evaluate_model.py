"""Run the 'evaluate' step of the pipeline. Usage: python scripts/evaluate_model.py [-v]"""

import sys

from metricmind.interfaces.cli.commands import run_command

if __name__ == "__main__":
    raise SystemExit(run_command("evaluate", sys.argv[1:]))
