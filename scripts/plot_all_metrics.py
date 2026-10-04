from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("data/processed/metrics.csv")
OUTPUT = Path("reports/figures/all_metrics_timeline.png")

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

metrics = [
    ("cpu_usage", "CPU Usage"),
    ("memory_usage", "Memory Usage"),
    ("memory_used", "Memory Used"),
    ("response_time_a", "Network Response Time A"),
    ("response_time_b", "Network Response Time B"),
    ("response_time_diff", "Network Response Time Difference"),
]

fig, axes = plt.subplots(
    6,
    1,
    figsize=(16, 14),
    sharex=True,
)

for ax, (column, title) in zip(axes, metrics):
    ax.plot(df["timestamp"], df[column], linewidth=0.8)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time")

fig.suptitle(
    "OS-Metrics: Six Metrics Synchronized Over Time",
    fontsize=16,
)

fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")

print(f"Saved: {OUTPUT}")