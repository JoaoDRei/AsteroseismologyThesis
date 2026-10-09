#!/usr/bin/env python3
"""Plot distributions from Baseline/manifest.csv for 80d entries only."""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot nu_max, delta_nu, teff, and evolstate distributions for manifest.csv 80d rows."
    )
    parser.add_argument(
        "--input",
        default=Path(__file__).with_name("manifest.csv"),
        help="Path to manifest.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).with_name("manifest_80d_plots"),
        help="Directory where individual plot images will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot(s) after saving.",
    )
    return parser.parse_args()


def read_manifest(path):
    df = pd.read_csv(path)
    required_columns = {"duration", "nu_max", "delta_nu", "teff", "evolstate"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing required columns: {', '.join(sorted(missing))}")

    df_80d = df[df["duration"].astype(str).str.strip() == "80d"].copy()
    if df_80d.empty:
        raise ValueError("No rows with duration == 80d were found in the manifest.csv file.")

    df_80d["nu_max"] = pd.to_numeric(df_80d["nu_max"], errors="coerce")
    df_80d["delta_nu"] = pd.to_numeric(df_80d["delta_nu"], errors="coerce")
    df_80d["teff"] = pd.to_numeric(df_80d["teff"], errors="coerce")
    df_80d["evolstate"] = df_80d["evolstate"].astype(str).fillna("Unknown")
    return df_80d


def plot_histogram(series, title, xlabel, output_path, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(series.dropna(), bins=30, color=color, edgecolor="#ffffff")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of stars")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot to: {output_path.resolve()}")


def plot_bar(series, title, xlabel, output_path, color):
    counts = series.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts.index, counts.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of stars")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot to: {output_path.resolve()}")


def make_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_histogram(
        df["nu_max"],
        r"Distribution of $\nu_{\mathrm{max}}$ (80d)",
        r"$\nu_{\mathrm{max}}$ [$\mu$Hz]",
        output_dir / "nu_max_distribution_80d.png",
        color="#2c7fb8",
    )
    plot_histogram(
        df["delta_nu"],
        r"Distribution of $\Delta\nu$ (80d)",
        r"$\Delta\nu$ [$\mu$Hz]",
        output_dir / "delta_nu_distribution_80d.png",
        color="#7fcdbb",
    )
    plot_histogram(
        df["teff"],
        r"Distribution of $T_{\mathrm{eff}}$ (80d)",
        r"$T_{\mathrm{eff}}$ [K]",
        output_dir / "teff_distribution_80d.png",
        color="#fdae61",
    )
    evolstate_filtered = df.loc[df["evolstate"] != "RC/RGB", "evolstate"]
    plot_bar(
        evolstate_filtered,
        "Distribution of evolstate (80d)",
        "evolstate",
        output_dir / "evolstate_distribution_80d.png",
        color="#ef3b2c",
    )


def main():
    args = parse_args()
    manifest_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Input manifest: {manifest_path.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    df_80d = read_manifest(manifest_path)
    make_plots(df_80d, output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
