"""Entry point:  ``metricmind <prepare|eda|train|evaluate>``  (or ``python -m metricmind``)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from metricmind.interfaces.cli.commands import COMMANDS, run_command


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        print(f"usage: metricmind {{{'|'.join(COMMANDS)}}} [-v]", file=sys.stderr)
        return 2
    return run_command(args[0], args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
