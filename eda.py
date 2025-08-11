"""
EDA: prints basic stats and saves simple plots under artifacts/plots/
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_wind_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/tibau_wind_data.csv")
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--decimal", type=str, default=".")
    args = ap.parse_args()

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_wind_csv(args.csv, sep=args.sep, decimal=args.decimal)

    print("Rows:", len(df))
    print("Time span:", df["timestamp"].min(), "to", df["timestamp"].max())
    print(df.describe(include="all"))

    # Plot wind_speed over time
    plt.figure()
    plt.plot(df["timestamp"], df["wind_speed"])
    plt.title("Wind Speed Over Time")
    plt.xlabel("Time")
    plt.ylabel("Wind Speed")
    plt.tight_layout()
    plt.savefig(out_dir / "wind_speed_over_time.png")

    # Histogram
    plt.figure()
    df["wind_speed"].hist(bins=40)
    plt.title("Wind Speed Distribution")
    plt.xlabel("Wind Speed")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "wind_speed_hist.png")

    # Direction histogram (if present)
    if "wind_direction" in df.columns:
        plt.figure()
        df["wind_direction"].hist(bins=36)
        plt.title("Wind Direction Distribution")
        plt.xlabel("Direction (deg)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "wind_direction_hist.png")

    print("Saved plots to", out_dir)

if __name__ == "__main__":
    main()
