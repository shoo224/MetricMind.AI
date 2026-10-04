"""MatplotlibFigureRenderer: report figures (headless 'Agg' backend, works on any machine)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from metricmind.application.ports import FigureRenderer  # noqa: E402

logger = logging.getLogger(__name__)


def _segments(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Contiguous True runs as (start_idx, end_idx) inclusive."""
    padded = np.concatenate(([False], mask, [False])).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


class MatplotlibFigureRenderer(FigureRenderer):
    def __init__(self, figures_dir: Path, dpi: int = 130) -> None:
        self._dir = figures_dir
        self._dpi = dpi

    def metric_timeseries(
        self, frame: pd.DataFrame, columns: Sequence[str], title: str, filename: str
    ) -> Path:
        fig, axes = plt.subplots(len(columns), 1, figsize=(12, 3 * len(columns)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, columns, strict=True):
            ax.plot(frame.index, frame[col], linewidth=0.6)
            ax.set_ylabel(col)
            self._shade_truth(ax, frame)
        axes[0].set_title(title)
        axes[-1].set_xlabel("time")
        if "is_anomaly" in frame.columns:
            axes[0].plot(
                [], [], color="orange", alpha=0.35, linewidth=8, label="labelled fault window"
            )
            axes[0].legend(loc="upper right")
        return self._save(fig, filename)

    def correlation_heatmap(
        self, frame: pd.DataFrame, columns: Sequence[str], filename: str
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            frame[list(columns)].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
        )
        ax.set_title("Pearson correlation between metrics")
        return self._save(fig, filename)

    def confusion_matrix(self, matrix: Sequence[Sequence[int]], title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(
            np.asarray(matrix),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        return self._save(fig, filename)

    def anomaly_overlay(
        self,
        frame: pd.DataFrame,
        metric: str,
        predictions: NDArray[np.int_],
        title: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frame.index, frame[metric], linewidth=0.6, label=metric)
        self._shade_truth(ax, frame)
        hit = predictions == 1
        ax.scatter(
            frame.index[hit],
            frame[metric][hit],
            s=8,
            color="tab:red",
            zorder=3,
            label="detected anomaly",
        )
        ax.plot([], [], color="orange", alpha=0.35, linewidth=8, label="labelled fault window")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel(metric)
        ax.legend(loc="upper right")
        return self._save(fig, filename)

    def score_timeline(
        self,
        frame: pd.DataFrame,
        scores: NDArray[np.float64],
        threshold: float,
        title: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(frame.index, scores, linewidth=0.5)
        ax.axhline(threshold, color="tab:red", linestyle="--", label=f"threshold = {threshold:.3f}")
        self._shade_truth(ax, frame)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("anomaly score (higher = more anomalous)")
        ax.legend(loc="upper right")
        return self._save(fig, filename)

    @staticmethod
    def _shade_truth(ax: plt.Axes, frame: pd.DataFrame) -> None:
        if "is_anomaly" not in frame.columns:
            return
        for start, end in _segments(frame["is_anomaly"].to_numpy() == 1):
            ax.axvspan(frame.index[start], frame.index[end], color="orange", alpha=0.25, lw=0)

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self._dpi)
        plt.close(fig)
        logger.info("Saved figure %s", path)
        return path
