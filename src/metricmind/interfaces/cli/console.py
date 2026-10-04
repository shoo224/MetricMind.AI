"""Console presentation for the CLI (the only place that prints)."""

from __future__ import annotations

import contextlib
import sys

from metricmind.application.ports import ProgressReporter

WIDTH = 60


def _make_stdout_safe() -> None:
    """Windows consoles may default to a legacy codepage that cannot print the check mark."""
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def banner(title: str) -> None:
    _make_stdout_safe()
    print("=" * WIDTH)
    print(f"MetricMind AI - {title}")
    print("=" * WIDTH)


def footer(message: str) -> None:
    print("\n" + "=" * WIDTH)
    print(message)
    print("=" * WIDTH)


class ConsoleReporter(ProgressReporter):
    def section(self, title: str) -> None:
        print(f"\n{title}...")

    def step(self, message: str) -> None:
        print(f"  ✓ {message}")
