"""Run the 'eda' step of the pipeline. Usage: python scripts/run_eda.py [-v]"""

import sys

from metricmind.interfaces.cli.commands import run_command

if __name__ == "__main__":
    raise SystemExit(run_command("eda", sys.argv[1:]))
