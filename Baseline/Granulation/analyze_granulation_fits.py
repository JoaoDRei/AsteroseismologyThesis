
#!/usr/bin/env python3
"""
Quality-control diagnostics for granulation fit results from v1 or v2.

Run:
    python analyze_granulation_fits.py

By default, analyzes the full-sample v2 results. An alternate fit-results CSV
can be supplied as the first argument. The analyzer does not refit spectra.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMS = [
    "long_component_height",
    "long_component_nu_char_uHz",
    "long_component_tau_days",
    "long_component_alpha",
    "short_component_height",
    "short_component_nu_char_uHz",
    "short_component_tau_days",
    "short_component_alpha",
    "white_noise",
    "oscillation_envelope_amplitude",
    "oscillation_envelope_centre_uHz",
    "oscillation_envelope_log_width",
    "rms_log10_residual",
]

POSITIVE = [
    p for p in PARAMS
    if "alpha" not in p and p != "rms_log10_residual"
]

BOUND_TOL = 0.01
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "fitted_parameters_v2_all_free_alpha"
    / "fit_results.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "granulation_v2_qc_all_free_alpha"
)
DEFAULT_ALPHA_BOUNDS = (2.0, 6.0)
DEFAULT_V1_ENVELOPE_WIDTH_BOUNDS = (0.05, 1.5)
DEFAULT_V2_ENVELOPE_WIDTH_BOUNDS = (0.08, 0.8)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def duration_from_name(name):
    m = re.search(r"-(20|55|80)d(?:_|\b)", str(name))
    return int(m.group(1)) if m else np.nan


def integrated_variance(height, char_freq, alpha):
    """
    Integral from 0 to infinity of:

        H / [1 + (nu / nu_char)^alpha] dnu

    = H * nu_char * pi / [alpha * sin(pi / alpha)]

    Units: native PSD units * microHz.

    Requires alpha > 1.

    This is not necessarily an observationally
    calibrated variance.
    """
    h = np.asarray(height, float)
    n = np.asarray(char_freq, float)
    a = np.asarray(alpha, float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):
        return h * n * np.pi / (
            a * np.sin(np.pi / a)
        )


def describe_numeric(series):
    s = pd.to_numeric(
        series, errors="coerce"
    ).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if s.empty:
        return {"n": 0}

    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_hist(df, col, path, log=False, boundaries=None):
    x = pd.to_numeric(
        df[col], errors="coerce"
    ).to_numpy(float)

    x = x[np.isfinite(x)]

    if log:
        x = x[x > 0]
        x = np.log10(x)

    if not len(x):
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        x,
        bins=65,
        color="steelblue",
        alpha=0.85,
    )

    if boundaries:
        for b in boundaries:
            ax.axvline(
                np.log10(b) if log else b,
                color="darkred",
                ls="--",
                lw=1,
            )

    ax.set(
        xlabel=("log10 " if log else "") + col,
        ylabel="Number of fits",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_duration_scatter(piv, col, a, b, path):
    if a not in piv.columns or b not in piv.columns:
        return

    sub = piv[[a, b]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    sub = sub[
        (sub[a] > 0) &
        (sub[b] > 0)
    ]

    if len(sub) < 3:
        return

    x = np.log10(sub[a].to_numpy(float))
    y = np.log10(sub[b].to_numpy(float))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.hexbin(
        x,
        y,
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    ax.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        lw=1,
    )

    ax.set(
        xlabel=f"log10 {col} ({a}d)",
        ylabel=f"log10 {col} ({b}d)",
        title=f"{len(sub)} matched KICs",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_FIT_RESULTS,
        help=(
            "Path to fit_results.csv "
            f"(default: {DEFAULT_FIT_RESULTS})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--bound-tolerance",
        type=float,
        default=BOUND_TOL,
        help="Absolute tolerance for alpha and width bounds",
    )
    parser.add_argument(
        "--alpha-bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override alpha bounds (otherwise read from run_config.json)",
    )
    parser.add_argument(
        "--envelope-width-bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override envelope log-width bounds (otherwise read from run_config.json or infer from schema)",
    )

    args = parser.parse_args()

    if args.bound_tolerance <= 0:
        parser.error(
            "--bound-tolerance must be positive"
        )

    df = pd.read_csv(
        args.csv,
        low_memory=False,
    )

    config_path = args.csv.parent / "run_config.json"
    run_config = (
        json.loads(config_path.read_text())
        if config_path.is_file()
        else {}
    )
    is_v2 = "nu_max_catalogue" in df.columns or "fit_alpha" in run_config

    configured_alpha_bounds = run_config.get("alpha_bounds")
    if not (
        isinstance(configured_alpha_bounds, list)
        and len(configured_alpha_bounds) == 2
    ):
        configured_alpha_bounds = DEFAULT_ALPHA_BOUNDS

    configured_width_bounds = run_config.get("envelope_width_bounds")
    if not (
        isinstance(configured_width_bounds, list)
        and len(configured_width_bounds) == 2
    ):
        configured_width_bounds = (
            DEFAULT_V2_ENVELOPE_WIDTH_BOUNDS
            if is_v2
            else DEFAULT_V1_ENVELOPE_WIDTH_BOUNDS
        )

    alpha_bounds = tuple(
        args.alpha_bounds
        if args.alpha_bounds is not None
        else configured_alpha_bounds
    )
    envelope_width_bounds = tuple(
        args.envelope_width_bounds
        if args.envelope_width_bounds is not None
        else configured_width_bounds
    )

    if alpha_bounds[0] <= 1 or alpha_bounds[0] >= alpha_bounds[1]:
        parser.error("--alpha-bounds must be increasing and greater than 1")
    if envelope_width_bounds[0] <= 0 or envelope_width_bounds[0] >= envelope_width_bounds[1]:
        parser.error("--envelope-width-bounds must be positive and increasing")

    if args.output_dir is not None:
        out = args.output_dir
    elif args.csv == DEFAULT_FIT_RESULTS:
        out = DEFAULT_OUTPUT_DIR
    else:
        out = args.csv.parent / "quality_control"

    out.mkdir(parents=True, exist_ok=True)

    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    required = set(PARAMS) | {
        "file",
        "kic",
        "success",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["kic"] = pd.to_numeric(
        df["kic"], errors="coerce"
    ).astype("Int64")

    df["duration_days"] = (
        df["file"]
        .map(duration_from_name)
        .astype("Int64")
    )

    for col in PARAMS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["success_bool"] = (
        df["success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # ----------------------------------------------
    # Parameter boundary diagnostics
    # ----------------------------------------------

    t = args.bound_tolerance
    alpha_min, alpha_max = alpha_bounds
    width_min, width_max = envelope_width_bounds

    for comp in ["long", "short"]:

        alpha = df[f"{comp}_component_alpha"]

        df[f"{comp}_alpha_at_lower"] = (
            (alpha - alpha_min).abs() <= t
        )

        df[f"{comp}_alpha_at_upper"] = (
            (alpha - alpha_max).abs() <= t
        )

        df[f"{comp}_alpha_at_bound"] = (
            df[f"{comp}_alpha_at_lower"]
            | df[f"{comp}_alpha_at_upper"]
        )

        h = df[f"{comp}_component_height"]
        n = df[f"{comp}_component_nu_char_uHz"]

        v = integrated_variance(h, n, alpha)

        df[
            f"{comp}_integrated_variance_native_uHz"
        ] = v

        df[
            f"{comp}_integrated_rms_native_sqrt_uHz"
        ] = np.sqrt(
            np.where(v >= 0, v, np.nan)
        )

    w = df["oscillation_envelope_log_width"]

    df["envelope_width_at_lower"] = (
        (w - width_min).abs() <= t
    )

    df["envelope_width_at_upper"] = (
        (w - width_max).abs() <= t
    )

    df["envelope_width_at_bound"] = (
        df["envelope_width_at_lower"]
        | df["envelope_width_at_upper"]
    )

    df["any_alpha_at_bound"] = (
        df["long_alpha_at_bound"]
        | df["short_alpha_at_bound"]
    )

    df["any_flagged_bound"] = (
        df["any_alpha_at_bound"]
        | df["envelope_width_at_bound"]
    )

    # ----------------------------------------------
    # Basic validity checks
    # ----------------------------------------------

    df["missing_or_invalid_parameter"] = (
        ~np.isfinite(
            df[PARAMS].to_numpy(float)
        ).all(axis=1)
    )

    df["invalid_positive_parameter"] = (
        (df[POSITIVE] <= 0).any(axis=1)
    )

    df["frequency_order_invalid"] = (
        df["long_component_nu_char_uHz"]
        > df["short_component_nu_char_uHz"]
    )

    df["both_char_frequencies_nearly_equal"] = (
        df["short_component_nu_char_uHz"]
        / df["long_component_nu_char_uHz"]
        < 1.2
    )

    df["envelope_near_short_component"] = (
        np.abs(
            np.log(
                df["oscillation_envelope_centre_uHz"]
                / df["short_component_nu_char_uHz"]
            )
        )
        < np.log(1.25)
    )

    df["usable_basic"] = (
        df["success_bool"]
        & ~df["missing_or_invalid_parameter"]
        & ~df["invalid_positive_parameter"]
        & ~df["frequency_order_invalid"]
    )

    df.to_csv(
        out / "fit_results_with_qc.csv",
        index=False,
    )

    # ----------------------------------------------
    # Summary statistics
    # ----------------------------------------------

    flag_cols = [
        "success_bool",
        "usable_basic",
        "missing_or_invalid_parameter",
        "invalid_positive_parameter",
        "long_alpha_at_lower",
        "long_alpha_at_upper",
        "short_alpha_at_lower",
        "short_alpha_at_upper",
        "any_alpha_at_bound",
        "envelope_width_at_lower",
        "envelope_width_at_upper",
        "any_flagged_bound",
        "both_char_frequencies_nearly_equal",
        "envelope_near_short_component",
    ]

    summary = {
        "total_rows": int(len(df)),
        "unique_kics": int(df["kic"].nunique()),
        "unknown_duration_rows": int(
            df["duration_days"].isna().sum()
        ),
        "duplicate_kic_duration_rows": int(
            df.duplicated(
                ["kic", "duration_days"]
            ).sum()
        ),
        "bound_tolerance": t,
        "input_csv": str(args.csv.resolve()),
        "fit_version": "v2" if is_v2 else "v1_or_legacy",
        "alpha_bounds": [alpha_min, alpha_max],
        "envelope_width_bounds": [width_min, width_max],
        "flags": {},
        "parameters": {},
        "by_duration": {},
        "cross_duration": {},
    }

    for c in flag_cols:
        summary["flags"][c] = {
            "n": int(df[c].sum()),
            "fraction": float(df[c].mean()),
        }

    for c in PARAMS + [
        "long_integrated_rms_native_sqrt_uHz",
        "short_integrated_rms_native_sqrt_uHz",
    ]:
        summary["parameters"][c] = (
            describe_numeric(df[c])
        )

    # ----------------------------------------------
    # Statistics separated by filter duration
    # ----------------------------------------------

    for dur, g in df.groupby(
        "duration_days",
        dropna=False,
    ):

        key = (
            "unknown"
            if pd.isna(dur)
            else str(int(dur))
        )

        summary["by_duration"][key] = {
            "rows": int(len(g)),
            "unique_kics": int(g["kic"].nunique()),
            "median_rms_log10": float(
                g["rms_log10_residual"].median()
            ),
            "fraction_any_alpha_bound": float(
                g["any_alpha_at_bound"].mean()
            ),
            "fraction_any_flagged_bound": float(
                g["any_flagged_bound"].mean()
            ),
        }

    df.groupby(
        "duration_days",
        dropna=False,
    )[flag_cols].agg(
        ["sum", "mean"]
    ).to_csv(
        out / "flags_by_duration.csv"
    )

    # ----------------------------------------------
    # Duplicate KIC-duration diagnostics
    # ----------------------------------------------

    duplicate_rows = df[
        df.duplicated(
            ["kic", "duration_days"],
            keep=False,
        )
    ].copy()

    duplicate_rows.to_csv(
        out / "duplicate_kic_duration_rows.csv",
        index=False,
    )

    # For cross-duration comparisons, retain
    # the lowest-RMS fit if duplicates exist.

    dedup = (
        df.dropna(
            subset=["kic", "duration_days"]
        )
        .sort_values(
            "rms_log10_residual",
            na_position="last",
        )
        .drop_duplicates(
            ["kic", "duration_days"]
        )
    )

    # ----------------------------------------------
    # Cross-duration stability
    # ----------------------------------------------

    metrics = [
        "long_component_height",
        "long_component_nu_char_uHz",
        "short_component_height",
        "short_component_nu_char_uHz",
        "long_integrated_rms_native_sqrt_uHz",
        "short_integrated_rms_native_sqrt_uHz",
        "oscillation_envelope_centre_uHz",
        "white_noise",
    ]

    stability_rows = []

    pairs = [
        (20, 55),
        (20, 80),
        (55, 80),
    ]

    for metric in metrics:

        piv = dedup.pivot(
            index="kic",
            columns="duration_days",
            values=metric,
        )

        for a, b in pairs:

            if (
                a not in piv.columns
                or b not in piv.columns
            ):
                continue

            sub = piv[[a, b]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            sub = sub[
                (sub[a] > 0)
                & (sub[b] > 0)
            ]

            if len(sub) == 0:
                continue

            ratios = sub[b] / sub[a]
            dex = np.log10(ratios)

            stability_rows.append({
                "parameter": metric,
                "duration_a": a,
                "duration_b": b,
                "n_matched": len(sub),
                "median_ratio_b_over_a": float(
                    np.median(ratios)
                ),
                "median_abs_log10_ratio": float(
                    np.median(np.abs(dex))
                ),
                "fraction_change_gt_25pct": float(
                    np.mean(
                        (ratios > 1.25)
                        | (ratios < 0.8)
                    )
                ),
                "fraction_change_gt_factor_2": float(
                    np.mean(
                        (ratios > 2)
                        | (ratios < 0.5)
                    )
                ),
                "spearman_r": float(
                    sub[a].corr(
                        sub[b],
                        method="spearman",
                    )
                ),
            })

            if metric in [
                "long_component_nu_char_uHz",
                "short_component_nu_char_uHz",
                "long_integrated_rms_native_sqrt_uHz",
                "short_integrated_rms_native_sqrt_uHz",
            ]:
                plot_duration_scatter(
                    piv,
                    metric,
                    a,
                    b,
                    plots / (
                        f"{metric}_{a}d_vs_{b}d.png"
                    ),
                )

    stability = pd.DataFrame(stability_rows)

    stability.to_csv(
        out / "duration_stability.csv",
        index=False,
    )

    if not stability.empty:

        summary["cross_duration"] = {
            (
                f"{r.parameter}_"
                f"{int(r.duration_a)}v"
                f"{int(r.duration_b)}"
            ): {
                "n": int(r.n_matched),
                "median_abs_log10_ratio": float(
                    r.median_abs_log10_ratio
                ),
                "fraction_change_gt_factor_2": float(
                    r.fraction_change_gt_factor_2
                ),
            }
            for r in stability.itertuples(index=False)
        }

    # ----------------------------------------------
    # Rank stars by cross-duration instability
    # ----------------------------------------------

    ranking = []

    for kic, g in dedup.groupby("kic"):

        if len(g) < 2:
            continue

        item = {
            "kic": int(kic),
            "n_durations": int(len(g)),
            "durations": ",".join(
                map(
                    str,
                    sorted(
                        g["duration_days"].astype(int)
                    ),
                )
            ),
            "median_rms_log10": float(
                g["rms_log10_residual"].median()
            ),
            "fraction_alpha_at_bound": float(
                g["any_alpha_at_bound"].mean()
            ),
        }

        vals = []

        for metric in [
            "long_component_nu_char_uHz",
            "short_component_nu_char_uHz",
            "long_integrated_rms_native_sqrt_uHz",
            "short_integrated_rms_native_sqrt_uHz",
        ]:

            x = g[metric].to_numpy(float)

            x = x[
                np.isfinite(x)
                & (x > 0)
            ]

            span = (
                float(
                    np.log10(
                        x.max() / x.min()
                    )
                )
                if len(x) >= 2
                else np.nan
            )

            item[
                f"{metric}_log10_span"
            ] = span

            if np.isfinite(span):
                vals.append(span)

        item["worst_log10_span"] = (
            max(vals) if vals else np.nan
        )

        ranking.append(item)

    ranking_df = pd.DataFrame(ranking)

    if not ranking_df.empty:

        ranking_df.sort_values(
            "worst_log10_span",
            ascending=False,
        ).to_csv(
            out / "most_unstable_stars.csv",
            index=False,
        )

        ranking_df.sort_values(
            "worst_log10_span",
            ascending=True,
        ).to_csv(
            out / "most_stable_stars.csv",
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            out / "most_unstable_stars.csv",
            index=False,
        )

        pd.DataFrame().to_csv(
            out / "most_stable_stars.csv",
            index=False,
        )

    # ----------------------------------------------
    # Parameter distribution plots
    # ----------------------------------------------

    plot_settings = [
        ("rms_log10_residual", False, None),
        ("long_component_alpha", False, [alpha_min, alpha_max]),
        ("short_component_alpha", False, [alpha_min, alpha_max]),
        (
            "oscillation_envelope_log_width",
            False,
            [width_min, width_max],
        ),
        ("long_component_nu_char_uHz", True, None),
        ("short_component_nu_char_uHz", True, None),
        (
            "long_integrated_rms_native_sqrt_uHz",
            True,
            None,
        ),
        (
            "short_integrated_rms_native_sqrt_uHz",
            True,
            None,
        ),
    ]

    for col, log, bounds in plot_settings:

        plot_hist(
            df,
            col,
            plots / f"hist_{col}.png",
            log=log,
            boundaries=bounds,
        )

    # ----------------------------------------------
    # JSON summary
    # ----------------------------------------------

    with open(out / "summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=False,
        )

    # ----------------------------------------------
    # Human-readable report
    # ----------------------------------------------

    report = []

    report.append(
        f"Granulation fitting QC: "
        f"{len(df):,} rows, "
        f"{df.kic.nunique():,} KICs"
    )

    report.append(
        f"Unknown durations: "
        f"{summary['unknown_duration_rows']}; "
        f"duplicate KIC-duration: "
        f"{summary['duplicate_kic_duration_rows']}"
    )

    report.append("")
    report.append(
        "FIT FLAGS (fraction of all rows):"
    )

    for c in flag_cols:

        v = summary["flags"][c]

        report.append(
            f"  {c:40s} "
            f"{v['n']:7,d} "
            f"({100*v['fraction']:6.2f}%)"
        )

    report.append("")
    report.append("BY DURATION:")

    for dur, v in summary["by_duration"].items():

        report.append(
            f"  {dur}d: "
            f"{v['rows']:,} rows, "
            f"median RMS="
            f"{v['median_rms_log10']:.4f}, "
            f"alpha bound="
            f"{100*v['fraction_any_alpha_bound']:.1f}%"
        )

    report.append("")
    report.append(
        "CROSS-DURATION STABILITY "
        "(median |log10 ratio|, "
        "factor-2 disagreement):"
    )

    for r in stability_rows:

        report.append(
            f"  {r['parameter']:45s} "
            f"{r['duration_a']}d/"
            f"{r['duration_b']}d "
            f"n={r['n_matched']:5d} "
            f"median="
            f"{r['median_abs_log10_ratio']:.3f} dex "
            f"factor2="
            f"{100*r['fraction_change_gt_factor_2']:.1f}%"
        )

    report.append("")
    report.append("INTERPRETATION WARNINGS:")

    report.append(
        "  - A successful optimizer exit is not "
        "evidence of an identifiable physical "
        "decomposition."
    )

    report.append(
        "  - Boundary flags use an absolute "
        "tolerance, not parameter uncertainties."
    )

    report.append(
        "  - Integrated RMS is "
        "sqrt(integral_0^infinity "
        "H/(1+(nu/nuchar)^alpha) dnu)."
    )

    report.append(
        "  - Its units are "
        "sqrt(native PSD power units * microHz); "
        "not automatically ppm."
    )

    report.append(
        "  - This integral extrapolates "
        "outside the observed frequency band."
    )

    report.append(
        "  - 20d/55d/80d are differently "
        "filtered PSDs; differences are not "
        "pure repeatability errors."
    )

    report.append(
        "  - Stability ranking is exploratory, "
        "not a calibrated acceptance threshold."
    )

    report.append(
        "  - PSD medians and log-space "
        "soft-L1 residuals are not "
        "a periodogram likelihood."
    )

    (out / "report.txt").write_text(
        "\n".join(report) + "\n"
    )

    print("\n".join(report))

    print(
        f"\nSaved outputs to {out.resolve()}"
    )


if __name__ == "__main__":
    main()
