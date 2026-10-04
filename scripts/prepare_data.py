"""Run the 'prepare' step of the pipeline. Usage: python scripts/prepare_data.py [-v]"""

import sys

from metricmind.interfaces.cli.commands import run_command

if __name__ == "__main__":
    raise SystemExit(run_command("prepare", sys.argv[1:]))
