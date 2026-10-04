"""CLI commands. Thin: parse args, call a use case, print a summary, map errors to exit codes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from metricmind.domain.exceptions import MetricMindError
from metricmind.interfaces.cli import console, wiring


def _run(title: str, body: Callable[[], str], verbose: bool) -> int:
    console.banner(title)
    try:
        closing = body()
    except MetricMindError as exc:
        print(f"\n✗ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    console.footer(closing)
    return 0


def prepare(verbose: bool = False) -> int:
    settings, spec = wiring.bootstrap(verbose)

    def body() -> str:
        s = wiring.build_prepare(settings, spec, console.ConsoleReporter()).execute()
        print(f"\nDataset:\n  Rows: {s['rows']:,}\n  Features: {len(s['metric_columns'])}")
        return f"Saved processed data to {settings.data_processed_dir}"

    return _run("Data Preparation", body, verbose)


def eda(verbose: bool = False) -> int:
    settings, _ = wiring.bootstrap(verbose)

    def body() -> str:
        s = wiring.build_eda(settings, console.ConsoleReporter()).execute()
        for a, b, r in s["highly_correlated_pairs"]:
            print(f"  ! {a} and {b} are highly correlated (r = {r:.3f}) -> redundant for modelling")
        return f"Figures saved to {settings.figures_dir}"

    return _run("Exploratory Data Analysis", body, verbose)


def train(verbose: bool = False) -> int:
    settings, _ = wiring.bootstrap(verbose)
    return _run(
        "Model Training",
        lambda: (
            wiring.build_train(settings, console.ConsoleReporter()).execute(),
            f"Model saved to {settings.model_dir}",
        )[1],
        verbose,
    )


def evaluate(verbose: bool = False) -> int:
    settings, _ = wiring.bootstrap(verbose)
    return _run(
        "Model Evaluation",
        lambda: (
            wiring.build_evaluate(settings, console.ConsoleReporter()).execute(),
            f"Results in {settings.results_dir}; figures in {settings.figures_dir}",
        )[1],
        verbose,
    )


COMMANDS: dict[str, Callable[[bool], int]] = {
    "prepare": prepare,
    "eda": eda,
    "train": train,
    "evaluate": evaluate,
}


def run_command(name: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"metricmind {name}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show INFO logs on the console"
    )
    args = parser.parse_args(argv)
    return COMMANDS[name](args.verbose)
