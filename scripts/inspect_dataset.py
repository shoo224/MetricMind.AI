from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/OSMetrics")

print("\n=== METRIC FILES ===")

metric_files = list(DATA_DIR.glob("*.csv"))

for file in metric_files:
    df = pd.read_csv(file)

    print(f"\nFile: {file.name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicate timestamps: {df['Time'].duplicated().sum()}")

    df["Time"] = pd.to_datetime(df["Time"])

    print(f"Start: {df['Time'].min()}")
    print(f"End:   {df['Time'].max()}")

    if len(df) > 1:
        intervals = df["Time"].diff().dropna()
        print(f"Most common sampling interval: {intervals.mode().iloc[0]}")

    print("First 3 rows:")
    print(df.head(3))

    print("Basic statistics:")
    print(df["Value"].describe())


print("\n=== EVENT LOGS ===")

event_dir = DATA_DIR / "event_logs"

for file in event_dir.glob("*.csv"):
    df = pd.read_csv(file)

    print(f"\nFile: {file.name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nEvent types:")
    print(df["type"].value_counts())

    print("\nEvent modes:")
    print(df["mode"].value_counts())

    print("\nStart point range:")
    print(df["start_point"].min(), "->", df["start_point"].max())

    print("End point range:")
    print(df["end_point"].min(), "->", df["end_point"].max())

    print("Start time:")
    print(df["start_time"].min())

    print("End time:")
    print(df["end_time"].max())
    