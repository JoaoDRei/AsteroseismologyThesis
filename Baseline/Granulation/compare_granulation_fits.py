#!/usr/bin/env python3
"""Compare v1 and catalogue-constrained v2 fits on matched KIC-duration PSDs.

Run from any directory:
    python compare_granulation_fits.py

The default comparison uses the full v1 results and the v2 results under
data/fitted_parameters_v2_all_free_alpha. It does not refit spectra.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1 = PROJECT_ROOT / "data" / "fitted_parameters" / "fit_results.csv"
DEFAULT_V2 = (
    PROJECT_ROOT
    / "data"
    / "fitted_parameters_v2_all_free_alpha"
    / "fit_results.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "granulation_comparison_all_free_alpha"

METRICS = [
    "rms_log10_residual",
    "long_component_height",
    "long_component_nu_char_uHz",
    "long_component_alpha",
    "long_integrated_rms_native_sqrt_uHz",
    "short_component_height",
    "short_component_nu_char_uHz",
    "short_component_alpha",
    "short_integrated_rms_native_sqrt_uHz",
    "white_noise",
    "oscillation_envelope_amplitude",
    "oscillation_envelope_centre_uHz",
    "oscillation_envelope_log_width",
]
STABILITY_METRICS = [
    "long_component_nu_char_uHz",
    "short_component_nu_char_uHz",
    "long_integrated_rms_native_sqrt_uHz",
    "short_integrated_rms_native_sqrt_uHz",
    "white_noise",
]
DURATIONS = (20, 55, 80)
PAIRS = ((20, 55), (20, 80), (55, 80))


def parse_duration(value):
    match = re.search(r"(?<!\d)(20|55|80)d(?!\d)", str(value), re.I)
    return int(match.group(1)) if match else np.nan


def parse_kic(value):
    match = re.search(r"kic0*(\d+)", str(value), re.I)
    return int(match.group(1)) if match else np.nan


def integrated_rms(height, nu, alpha):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = height * nu * np.pi / (alpha * np.sin(np.pi / alpha))
        return np.sqrt(np.where((alpha > 1) & (variance >= 0), variance, np.nan))


def spearman_correlation(a, b):
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b, method="spearman"))


def read_fits(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} fit results not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"{label} fit results contain no rows: {path}")

    required = {
        "file",
        "kic",
        "success",
        "rms_log10_residual",
    }
    for component in ("long", "short"):
        required.update({
            f"{component}_component_height",
            f"{component}_component_nu_char_uHz",
            f"{component}_component_alpha",
        })
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} results missing required columns: {missing}")

    df["kic"] = pd.to_numeric(df["kic"], errors="coerce")
    df["kic"] = df["kic"].fillna(df["file"].map(parse_kic))
    if "duration" in df.columns:
        df["duration_days"] = df["duration"].map(parse_duration)
        df["duration_days"] = df["duration_days"].fillna(
            df["file"].map(parse_duration)
        )
    else:
        df["duration_days"] = df["file"].map(parse_duration)

    df["success_bool"] = (
        df["success"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    )
    numeric_columns = set(METRICS) | {
        "long_component_alpha",
        "short_component_alpha",
        "long_component_height",
        "short_component_height",
        "long_component_nu_char_uHz",
        "short_component_nu_char_uHz",
    }
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    for component in ("long", "short"):
        df[f"{component}_integrated_rms_native_sqrt_uHz"] = integrated_rms(
            df[f"{component}_component_height"].to_numpy(float),
            df[f"{component}_component_nu_char_uHz"].to_numpy(float),
            df[f"{component}_component_alpha"].to_numpy(float),
        )

    df = df.loc[
        df["success_bool"]
        & df["kic"].notna()
        & df["duration_days"].isin(DURATIONS)
    ].copy()
    df["kic"] = df["kic"].astype("int64")
    df["duration_days"] = df["duration_days"].astype(int)

    duplicates = df.duplicated(["kic", "duration_days"], keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, ["kic", "duration_days", "file"]].head(10)
        raise ValueError(
            f"{label} has duplicate successful KIC-duration fits:\n"
            f"{examples.to_string(index=False)}"
        )
    return df


def comparison_summary(matched):
    rows = []
    for metric in METRICS:
        a = pd.to_numeric(matched[f"{metric}_v1"], errors="coerce")
        b = pd.to_numeric(matched[f"{metric}_v2"], errors="coerce")
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]
        if a.empty:
            continue

        positive = (a > 0) & (b > 0)
        log_ratio = np.log10(b[positive] / a[positive])
        rows.append({
            "metric": metric,
            "n_matched": int(len(a)),
            "v1_median": float(a.median()),
            "v2_median": float(b.median()),
            "median_v2_minus_v1": float((b - a).median()),
            "median_absolute_difference": float((b - a).abs().median()),
            "median_v2_over_v1": (
                float(np.median(b[positive] / a[positive]))
                if positive.any() else np.nan
            ),
            "median_absolute_log10_ratio": (
                float(np.median(np.abs(log_ratio)))
                if len(log_ratio) else np.nan
            ),
            "spearman_r": spearman_correlation(a, b),
        })
    return pd.DataFrame(rows)


def duration_stability(frames):
    rows = []
    for label, df in frames.items():
        for metric in STABILITY_METRICS:
            piv = df.pivot(index="kic", columns="duration_days", values=metric)
            for duration_a, duration_b in PAIRS:
                if duration_a not in piv.columns or duration_b not in piv.columns:
                    continue
                pair = piv[[duration_a, duration_b]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                pair = pair[(pair[duration_a] > 0) & (pair[duration_b] > 0)]
                if pair.empty:
                    continue
                ratio = pair[duration_b] / pair[duration_a]
                rows.append({
                    "version": label,
                    "metric": metric,
                    "duration_a": duration_a,
                    "duration_b": duration_b,
                    "n_matched_kics": int(len(pair)),
                    "median_ratio_b_over_a": float(ratio.median()),
                    "median_absolute_log10_ratio": float(
                        np.median(np.abs(np.log10(ratio)))
                    ),
                    "spearman_r": spearman_correlation(
                        pair[duration_a], pair[duration_b]
                    ),
                })
    return pd.DataFrame(rows)


def save_scatter_plots(matched, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        x = pd.to_numeric(matched[f"{metric}_v1"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(matched[f"{metric}_v2"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if not valid.any():
            continue
        x, y = x[valid], y[valid]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(x, y, s=8, alpha=0.35, rasterized=True)
        limits = (
            min(float(x.min()), float(y.min())),
            max(float(x.max()), float(y.max())),
        )
        ax.plot(limits, limits, "k--", lw=1, label="y = x")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=limits,
            ylim=limits,
            xlabel=f"V1 {metric}",
            ylabel=f"V2 {metric}",
            title=f"Matched fits (n={len(x):,})",
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_v1_vs_v2.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    v1 = read_fits(args.v1, "v1")
    v2 = read_fits(args.v2, "v2")
    key = ["kic", "duration_days"]
    v1_columns = key + ["file"] + [c for c in METRICS if c in v1.columns]
    v2_columns = key + ["file"] + [c for c in METRICS if c in v2.columns]
    v1_for_merge = v1[v1_columns].rename(
        columns={column: f"{column}_v1" for column in v1_columns if column not in key}
    )
    v2_for_merge = v2[v2_columns].rename(
        columns={column: f"{column}_v2" for column in v2_columns if column not in key}
    )
    matched = v1_for_merge.merge(
        v2_for_merge,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    if matched.empty:
        raise ValueError("No matching successful KIC-duration fits between v1 and v2.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output_dir / "matched_fits.csv", index=False)
    summary = comparison_summary(matched)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    stability = duration_stability({"v1": v1, "v2": v2})
    stability.to_csv(args.output_dir / "duration_stability.csv", index=False)
    save_scatter_plots(matched, args.output_dir / "plots")
    run_info = {
        "v1": str(args.v1.resolve()),
        "v2": str(args.v2.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "v1_successful_fits": int(len(v1)),
        "v2_successful_fits": int(len(v2)),
        "matched_kic_duration_rows": int(len(matched)),
        "matched_kics": int(matched["kic"].nunique()),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(
        f"Compared {len(matched):,} matched KIC-duration fits "
        f"({matched['kic'].nunique():,} KICs)."
    )
    print(f"Comparison outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
