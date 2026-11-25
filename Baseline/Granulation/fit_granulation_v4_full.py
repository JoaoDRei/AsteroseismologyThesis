#!/usr/bin/env python3
"""
V4 Harvey-background fitting for four-year KEPSEISMIC long-cadence PSDs.

Extends the V3 pipeline without changing its objective or preprocessing:
  - logarithmic frequency bins and median PSD power
  - soft-L1 least squares in log10 power
  - multiple starting points
  - interleaved train/held-out bins for diagnostics
  - final refit on all bins
  - periodic CSV saving, plots, and reproducible KIC sampling

Models:
  one_fixed   : 1 Harvey component, alpha=4
  two_fixed   : 2 Harvey components, alpha=4
  two_free    : 2 Harvey components, alpha in [2,6]
  three_fixed : 3 Harvey components, alpha=4

The three_fixed components are labeled low/mid/high in increasing
characteristic frequency. These are NOT automatically physical
supergranulation/mesogranulation/granulation identifications.

NOTE: 20d, 55d and 80d are KADACS filtering timescales, not
observing lengths. The PSDs are based on approximately four years
of Kepler observations at ~29.4-minute cadence.

This is an incremental V3 comparison, NOT an exact Corsaro et al.
background model: it does not add colored noise, the Kepler response
function, or the paper's linear-frequency Gaussian envelope.

Run from your usual script location:
    python fit_granulation_v4_full.py

All configuration is below. No CLI arguments.
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

# Assumes this script lives two levels below the project root,
# as in your original V3 script. Adjust if relocated.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "Baseline" / "manifest.csv"
INPUT_DIR = PROJECT_ROOT / "data" / "psd_physical"

# First compare on the same PSDs; do NOT overwrite V3 output.
MODELS = ("three_fixed",)
VALID_MODELS = ("one_fixed", "two_fixed", "two_free", "three_fixed")

# 0 = full catalogue, 300 = random sample of 300 unique KICs.
MAX_STARS = 0
SAMPLE_SEED = 42

# Use 8 for three-component exploratory fits. Can increase if
# convergence/stability diagnostics suggest local minima.
N_STARTS = 8
MAX_NFEV = 12000
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 0.15
HOLDOUT_EVERY = 5

MIN_FREQUENCY = 0.1
MAX_FREQUENCY = 283.0
N_LOG_BINS = 180
MIN_BIN_POINTS = 4
MIN_RAW_POINTS = 100
MIN_POPULATED_BINS = 30

OUTPUT_DIR = PROJECT_ROOT / "data" / "fitted_parameters_v4_three_fixed_all"
PLOT_DIR = PROJECT_ROOT / "results" / "granulation_v4_three_fixed_all_plots"
MAKE_PLOTS = True
PLOT_FRACTION = 0.15
PLOT_SEED = 42
SAVE_EVERY = 25
ALLOW_OVERWRITE = False

FIXED_ALPHA = 4.0
ALPHA_MIN = 2.0
ALPHA_MAX = 6.0
ENVELOPE_WIDTH_MIN = 0.08
ENVELOPE_WIDTH_MAX = 0.8
TINY = np.finfo(float).tiny

# These are STARTING GUESSES, not constraints on fitted frequencies.
TWO_START_RATIOS = (1 / 20, 1 / 3)
THREE_START_PATTERNS = (
    (1 / 20, 1 / 3, 1.0),
    (1 / 40, 1 / 4, 0.9),
    (1 / 10, 1 / 2, 1.5),
    (1 / 30, 1 / 3, 2.0),
)

# A conservative numerical flag, not a physical-quality guarantee.
BOUND_RELATIVE_TOLERANCE = 0.01


# ============================================================
# FILE AND MANIFEST UTILITIES
# ============================================================

def parse_filename(path: Path):
    """Extract KIC and KADACS filtering duration from PSD filename."""
    match = re.search(
        r"kic0*(\d+).*?(20|55|80)d",
        path.stem,
        flags=re.I,
    )
    if not match:
        raise ValueError(f"Cannot identify KIC/duration: {path.name}")
    return int(match.group(1)), f"{match.group(2)}d"


def select_plot_kics(files, fraction, seed):
    """Select KICs, not spectra, to preserve all duration examples."""
    available = np.array(
        sorted({kic for _, kic, _ in files}), dtype=np.int64
    )
    if fraction <= 0 or not len(available):
        return set()
    sample_size = min(
        len(available),
        max(1, int(round(len(available) * fraction))),
    )
    rng = np.random.default_rng(seed)
    return set(
        rng.choice(available, size=sample_size, replace=False).tolist()
    )


def load_manifest(path: Path, tolerance=0.01):
    """Read nu_max and valid (KIC, filtering duration) combinations."""
    df = pd.read_csv(path, usecols=["kic", "duration", "nu_max"])
    df["kic"] = pd.to_numeric(df["kic"], errors="raise").astype("int64")
    df["nu_max"] = pd.to_numeric(df["nu_max"], errors="coerce")
    df["duration"] = (
        df["duration"].astype(str).str.strip().str.lower()
    )
    df = df[np.isfinite(df["nu_max"]) & (df["nu_max"] > 0)].copy()
    if df.empty:
        raise ValueError("No positive finite nu_max in manifest.")

    grouped = df.groupby("kic")["nu_max"]
    inconsistent = grouped.agg(
        lambda x: (x.max() - x.min()) / x.median() > tolerance
    )
    if inconsistent.any():
        bad = inconsistent[inconsistent].index.tolist()[:15]
        raise ValueError(f"Inconsistent catalogue nu_max: {bad}")

    return grouped.median().to_dict(), set(
        zip(df["kic"], df["duration"])
    )


# ============================================================
# PSD BINNING (UNCHANGED V3 METHODOLOGY)
# ============================================================

def log_bin(freq, power, bins=N_LOG_BINS, min_points=MIN_BIN_POINTS):
    """Geometric-mean frequency and median PSD power per log bin."""
    edges = np.geomspace(freq.min(), freq.max(), bins + 1)
    indices = np.searchsorted(edges, freq, side="right") - 1
    indices = np.clip(indices, 0, bins - 1)

    xs, ys, counts = [], [], []
    for j in range(bins):
        mask = indices == j
        count = int(mask.sum())
        if count < min_points:
            continue
        xs.append(float(np.exp(np.mean(np.log(freq[mask])))))
        ys.append(float(np.median(power[mask])))
        counts.append(count)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(counts, dtype=int),
    )


# ============================================================
# PHYSICAL MODEL
# ============================================================

def harvey(nu, height, nu_char, alpha):
    """Harvey-like background term in native PSD units."""
    return height / (1.0 + (nu / nu_char) ** alpha)


def integrated_rms(height, nu_char, alpha):
    """
    sqrt(integral_0^infinity H/[1+(nu/nu_char)^alpha] dnu).

    A native-unit extrapolated amplitude proxy, not guaranteed ppm.
    """
    integral = (
        height * nu_char * np.pi
        / (alpha * np.sin(np.pi / alpha))
    )
    return float(np.sqrt(integral))


def number_of_components(model):
    return {
        "one_fixed": 1,
        "two_fixed": 2,
        "two_free": 2,
        "three_fixed": 3,
    }[model]


def unpack(theta, model):
    """
    Decode natural-log positive parameters, sort Harvey components
    in increasing characteristic frequency, and decode nuisance terms.
    """
    n = number_of_components(model)
    components = []
    offset = 0

    for _ in range(n):
        height = float(np.exp(theta[offset]))
        char = float(np.exp(theta[offset + 1]))
        if model == "two_free":
            alpha = float(theta[offset + 2])
            offset += 3
        else:
            alpha = FIXED_ALPHA
            offset += 2
        components.append((height, char, alpha))

    components.sort(key=lambda c: c[1])
    white, amplitude, width = (
        float(np.exp(v)) for v in theta[offset:offset + 3]
    )
    return components, white, amplitude, width


def evaluate(theta, nu, nu_max, model):
    """Total model and ordered component contributions."""
    components, white, amplitude, width = unpack(theta, model)
    parts = [harvey(nu, *component) for component in components]
    parts.append(np.full_like(nu, white, dtype=float))
    parts.append(
        amplitude * np.exp(
            -0.5 * (np.log(nu / nu_max) / width) ** 2
        )
    )
    return np.sum(parts, axis=0), parts


# ============================================================
# PARAMETER BOUNDS
# ============================================================

def make_bounds(xb, yb, model):
    """V3-style broad frequency and height bounds, no nu_max priors."""
    fmin = float(xb.min())
    fmax = float(xb.max())
    ymax = max(float(np.percentile(yb, 90)), TINY * 1e20)
    ymin = max(float(np.percentile(yb, 10) * 0.1), TINY * 1e20)

    log_h_lo = np.log(ymin * 1e-3)
    log_h_hi = np.log(ymax * 1e3)
    log_c_lo = np.log(fmin / 10)
    log_c_hi = np.log(fmax * 10)

    lower, upper = [], []
    for _ in range(number_of_components(model)):
        lower.extend([log_h_lo, log_c_lo])
        upper.extend([log_h_hi, log_c_hi])
        if model == "two_free":
            lower.append(ALPHA_MIN)
            upper.append(ALPHA_MAX)

    lower.extend([
        np.log(ymin * 1e-4),
        np.log(ymin * 1e-3),
        np.log(ENVELOPE_WIDTH_MIN),
    ])
    upper.extend([
        np.log(ymax * 10),
        np.log(ymax * 1e3),
        np.log(ENVELOPE_WIDTH_MAX),
    ])
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


# ============================================================
# INITIAL PARAMETERS
# ============================================================

def make_initial(xb, yb, nu_max, model, start, rng):
    """
    V3 starting values for one/two-component models; several
    physically motivated but unconstrained starts for three_fixed.
    """
    fmin = float(xb.min())
    fmax = float(xb.max())
    ymax = float(np.percentile(yb, 90))
    ymin = max(float(np.percentile(yb, 10) * 0.1), TINY * 1e20)

    if model == "one_fixed":
        frequencies = [nu_max / 3]
        heights = [max(ymax * 0.8, ymin)]
    elif model in ("two_fixed", "two_free"):
        frequencies = [nu_max / 20, nu_max / 3]
        heights = [max(ymax * 0.7, ymin), max(ymax * 0.3, ymin)]
    elif model == "three_fixed":
        pattern = THREE_START_PATTERNS[start % len(THREE_START_PATTERNS)]
        frequencies = [nu_max * ratio for ratio in pattern]
        heights = [
            max(ymax * fraction, ymin)
            for fraction in (0.55, 0.30, 0.15)
        ]
    else:
        raise ValueError(model)

    if start > 0:
        frequencies = [
            f * np.exp(rng.normal(0, 1.0)) for f in frequencies
        ]
        heights = [
            h * np.exp(rng.normal(0, 1.0)) for h in heights
        ]

    frequencies = [
        float(np.clip(f, fmin * 1.1, fmax / 1.1))
        for f in frequencies
    ]

    parameters = []
    for height, frequency in zip(heights, frequencies):
        parameters.extend([np.log(height), np.log(frequency)])
        if model == "two_free":
            alpha = (
                FIXED_ALPHA if start == 0
                else float(rng.uniform(2.2, 5.8))
            )
            parameters.append(alpha)

    white = (
        ymin if start == 0
        else ymin * np.exp(rng.normal(0, 0.5))
    )
    envelope = max(ymax * 0.3, ymin)
    if start > 0:
        envelope *= np.exp(rng.normal(0, 0.7))
    width = (
        0.35 if start == 0
        else float(rng.uniform(0.10, 0.65))
    )
    parameters.extend([
        np.log(white), np.log(envelope), np.log(width)
    ])
    return np.asarray(parameters, dtype=float)


# ============================================================
# FITTING
# ============================================================

def log_residual(theta, x, y, nu_max, model):
    """Predicted minus observed log10 PSD power."""
    prediction, _ = evaluate(theta, x, nu_max, model)
    return np.log10(np.maximum(prediction, TINY)) - np.log10(y)


def fit_variant(xb, yb, nu_max, model, rng):
    """
    V3 fitting protocol:
      1. Every fifth populated bin held out.
      2. Multiple starts fitted on training bins.
      3. Select lowest training robust cost among successful fits.
      4. Record training and held-out residuals.
      5. Refit all bins from selected training solution.

    Held-out bins are diagnostic, not independent measurements.
    """
    holdout = np.arange(len(xb)) % HOLDOUT_EVERY == 0
    training = ~holdout
    if training.sum() < 25 or holdout.sum() < 5:
        raise ValueError("Too few train/holdout bins.")

    lower, upper = make_bounds(xb, yb, model)
    candidates = []

    for start in range(N_STARTS):
        p0 = make_initial(xb, yb, nu_max, model, start, rng)
        p0 = np.clip(p0, lower + 1e-8, upper - 1e-8)
        try:
            fit = least_squares(
                log_residual,
                p0,
                args=(xb[training], yb[training], nu_max, model),
                bounds=(lower, upper),
                loss=ROBUST_LOSS,
                f_scale=ROBUST_F_SCALE,
                max_nfev=MAX_NFEV,
            )
            if (
                fit.success
                and np.all(np.isfinite(fit.x))
                and np.isfinite(fit.cost)
            ):
                candidates.append((float(fit.cost), start, fit))
        except (FloatingPointError, ValueError, OverflowError):
            continue

    if not candidates:
        raise RuntimeError(
            "No converged training fit from any initialization."
        )

    candidates.sort(key=lambda item: item[0])
    _, best_start, best_train = candidates[0]

    train_residuals = log_residual(
        best_train.x, xb[training], yb[training], nu_max, model
    )
    holdout_residuals = log_residual(
        best_train.x, xb[holdout], yb[holdout], nu_max, model
    )

    final = least_squares(
        log_residual,
        best_train.x,
        args=(xb, yb, nu_max, model),
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
        max_nfev=MAX_NFEV,
    )
    if not final.success:
        raise RuntimeError(
            "Full-data optimizer did not converge: " + str(final.message)
        )
    if not np.all(np.isfinite(final.x)) or not np.isfinite(final.cost):
        raise RuntimeError("Nonfinite full-data fit.")

    full_residuals = log_residual(
        final.x, xb, yb, nu_max, model
    )
    return {
        "fit": final,
        "n_successful_starts": len(candidates),
        "best_start": best_start,
        "train_rms_log10": float(np.sqrt(np.mean(train_residuals ** 2))),
        "holdout_rms_log10": float(np.sqrt(np.mean(holdout_residuals ** 2))),
        "full_rms_log10": float(np.sqrt(np.mean(full_residuals ** 2))),
        "n_train_bins": int(training.sum()),
        "n_holdout_bins": int(holdout.sum()),
        "train_success": True,
        "full_success": True,
        "train_message": str(best_train.message),
        "full_message": str(final.message),
    }


# ============================================================
# OUTPUT ROWS
# ============================================================

def result_row(path, kic, duration, nu_max, xb, freq, model, fit_info):
    """Keep V3 columns and append V4 low/mid/high diagnostics."""
    fit = fit_info["fit"]
    components, white, amplitude, width = unpack(fit.x, model)

    row = {
        "file": path.name,
        "kic": kic,
        "duration": duration,
        "nu_max_catalogue": nu_max,
        "model": model,
        "success": True,
        "cost": float(fit.cost),
        "rms_log10_residual": fit_info["full_rms_log10"],
        "train_rms_log10_residual": fit_info["train_rms_log10"],
        "holdout_rms_log10_residual": fit_info["holdout_rms_log10"],
        "n_successful_starts": fit_info["n_successful_starts"],
        "best_start": fit_info["best_start"],
        "n_raw_points": len(freq),
        "n_fit_bins": len(xb),
        "n_train_bins": fit_info["n_train_bins"],
        "n_holdout_bins": fit_info["n_holdout_bins"],
        "white_noise": white,
        "oscillation_envelope_amplitude": amplitude,
        "oscillation_envelope_centre_uHz": nu_max,
        "oscillation_envelope_log_width": width,
        "message": fit_info["full_message"],
    }

    # Explicitly initialize all component columns for a stable schema.
    for label in ("long", "short", "low", "mid", "high"):
        row.update({
            f"{label}_component_height": np.nan,
            f"{label}_component_nu_char_uHz": np.nan,
            f"{label}_component_tau_days": np.nan,
            f"{label}_component_alpha": np.nan,
            f"{label}_integrated_rms_native_sqrt_uHz": np.nan,
            f"{label}_alpha_at_bound": False,
            f"{label}_char_below_fit_band": False,
            f"{label}_char_below_filter_timescale": False,
            f"{label}_nu_char_over_nu_max": np.nan,
            f"{label}_char_near_parameter_bound": False,
        })

    if model == "one_fixed":
        labels = ("short",)
    elif model in ("two_fixed", "two_free"):
        labels = ("long", "short")
    else:
        labels = ("low", "mid", "high")

    # Frequency bound is independent of yb:
    frequency_min_bound = float(xb.min()) / 10
    frequency_max_bound = float(xb.max()) * 10
    filter_frequency = 1e6 / (int(duration[:-1]) * 86400)

    for label, (height, char, alpha) in zip(labels, components):
        row[f"{label}_component_height"] = height
        row[f"{label}_component_nu_char_uHz"] = char
        row[f"{label}_component_tau_days"] = (
            1.0 / (2 * np.pi * char * 1e-6 * 86400)
        )
        row[f"{label}_component_alpha"] = alpha
        row[f"{label}_integrated_rms_native_sqrt_uHz"] = (
            integrated_rms(height, char, alpha)
        )
        row[f"{label}_alpha_at_bound"] = bool(
            model == "two_free"
            and min(abs(alpha - ALPHA_MIN), abs(alpha - ALPHA_MAX)) <= 0.01
        )
        row[f"{label}_char_below_fit_band"] = bool(char < xb.min())
        row[f"{label}_char_below_filter_timescale"] = bool(
            char < filter_frequency
        )
        row[f"{label}_nu_char_over_nu_max"] = char / nu_max
        row[f"{label}_char_near_parameter_bound"] = bool(
            char <= frequency_min_bound * (1 + BOUND_RELATIVE_TOLERANCE)
            or char >= frequency_max_bound / (1 + BOUND_RELATIVE_TOLERANCE)
        )

    # These ratios are descriptive, NOT forced physical identifications.
    row.update({
        "mid_to_high_frequency_ratio": np.nan,
        "mid_to_high_amplitude_ratio": np.nan,
        "mid_near_nu_max_over_3_factor": np.nan,
        "high_near_nu_max_factor": np.nan,
        "low_to_mid_frequency_ratio": np.nan,
    })
    if model == "three_fixed":
        low, mid, high = components
        row["low_to_mid_frequency_ratio"] = low[1] / mid[1]
        row["mid_to_high_frequency_ratio"] = mid[1] / high[1]
        row["mid_to_high_amplitude_ratio"] = (
            integrated_rms(*mid) / integrated_rms(*high)
        )
        row["mid_near_nu_max_over_3_factor"] = mid[1] / (nu_max / 3)
        row["high_near_nu_max_factor"] = high[1] / nu_max

    return row


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_fit(
    path, kic, duration, nu_max, freq, power, xb, yb, results, output
):
    """
    Six-panel V3-style diagnostics:
      comparison, decomposition, two residual panels,
      two train/held-out panels.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 12), sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    dense = np.geomspace(freq.min(), freq.max(), 1000)

    ax = axes[0, 0]
    ax.loglog(
        freq, power, ".", alpha=0.10, ms=1,
        color="0.5", label="Raw PSD",
    )
    ax.loglog(xb, yb, ".", ms=4, color="black", label="Binned median")
    for model in MODELS:
        if model not in results:
            continue
        prediction, _ = evaluate(
            results[model]["fit"].x, dense, nu_max, model
        )
        ax.loglog(dense, prediction, lw=1.7, label=model)
    ax.axvline(
        nu_max, color="0.4", ls=":", label=r"Catalogue $\nu_{\max}$"
    )
    ax.set(
        title="Model comparison", xlabel="Frequency (microHz)",
        ylabel="PSD (native units)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    ax2 = axes[0, 1]
    chosen = next(
        (m for m in ("three_fixed", "two_fixed", "two_free", "one_fixed")
         if m in results)
    )
    prediction, parts = evaluate(
        results[chosen]["fit"].x, dense, nu_max, chosen
    )
    ax2.loglog(xb, yb, ".", ms=4, color="black", label="Binned median")
    ax2.loglog(dense, prediction, lw=2, label=chosen)
    if chosen == "three_fixed":
        names = ["Low Harvey", "Mid Harvey", "High Harvey"]
    elif chosen == "one_fixed":
        names = ["Harvey"]
    else:
        names = ["Long Harvey", "Short Harvey"]
    names += ["White noise", "Oscillation envelope"]
    for component, name in zip(parts, names):
        ax2.loglog(dense, component, "--", alpha=0.75, label=name)
    ax2.axvline(nu_max, color="0.4", ls=":")
    ax2.axvline(nu_max / 3, color="0.6", ls=":")
    ax2.set(
        title=f"{chosen} decomposition",
        xlabel="Frequency (microHz)",
        ylabel="PSD (native units)",
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_ylim(ax.get_ylim())

    for col in range(2):
        residual_ax = axes[1, col]
        for model in MODELS:
            if model not in results:
                continue
            residual = log_residual(
                results[model]["fit"].x, xb, yb, nu_max, model
            )
            residual_ax.semilogx(
                xb, -residual, ".", ms=3, label=model
            )
        residual_ax.axhline(0, color="black", lw=1)
        residual_ax.set(
            xlabel="Frequency (microHz)",
            ylabel="log10(data/fit)",
        )
        residual_ax.grid(True, which="both", alpha=0.2)

    for col in range(2):
        metric_ax = axes[2, col]
        for model in MODELS:
            if model not in results:
                continue
            info = results[model]
            metric_ax.plot(
                [0, 1],
                [info["train_rms_log10"], info["holdout_rms_log10"]],
                "o-", label=model,
            )
        metric_ax.set(
            xticks=[0, 1], xticklabels=["Train", "Held out"],
            ylabel="log10 RMS (dex)",
        )
        metric_ax.grid(True, alpha=0.2)
    axes[2, 0].legend(fontsize=8)

    fig.suptitle(
        f"KIC {kic} | {duration} | "
        f"$\\nu_{{\\max}}$={nu_max:.2f} microHz"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_tables(rows, failures, output_dir):
    """Atomic replacement of V3-style fit and failure CSVs."""
    results_path = output_dir / "fit_results.csv"
    failures_path = output_dir / "fit_failures.csv"
    results_temp = output_dir / "fit_results.tmp"
    failures_temp = output_dir / "fit_failures.tmp"

    pd.DataFrame(rows).to_csv(results_temp, index=False)
    pd.DataFrame(
        failures,
        columns=["file", "kic", "duration", "model", "error"],
    ).to_csv(failures_temp, index=False)

    results_temp.replace(results_path)
    failures_temp.replace(failures_path)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def write_summary(rows, output_dir):
    """Model summaries and paired holdout comparisons."""
    df = pd.DataFrame(rows)
    if df.empty:
        (output_dir / "model_comparison.csv").write_text(
            "model,n,median_full_rms,median_holdout_rms\n"
        )
        return

    statistics = []
    for model, part in df.groupby("model"):
        if model == "three_fixed":
            primary_label = "low"
        else:
            primary_label = "long"
        statistics.append({
            "model": model,
            "n": len(part),
            "median_full_rms": part["rms_log10_residual"].median(),
            "median_train_rms": part[
                "train_rms_log10_residual"
            ].median(),
            "median_holdout_rms": part[
                "holdout_rms_log10_residual"
            ].median(),
            "long_char_below_fit_band_fraction": part[
                "long_char_below_fit_band"
            ].mean(),
            "long_char_below_filter_timescale_fraction": part[
                "long_char_below_filter_timescale"
            ].mean(),
            "long_alpha_at_bound_fraction": part[
                "long_alpha_at_bound"
            ].mean(),
            "lowest_char_below_fit_band_fraction": part[
                f"{primary_label}_char_below_fit_band"
            ].mean(),
            "lowest_char_below_filter_timescale_fraction": part[
                f"{primary_label}_char_below_filter_timescale"
            ].mean(),
            "lowest_char_near_parameter_bound_fraction": part[
                f"{primary_label}_char_near_parameter_bound"
            ].mean(),
        })

    pd.DataFrame(statistics).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    wide = df.pivot_table(
        index=["kic", "duration"], columns="model",
        values="holdout_rms_log10_residual", aggfunc="first",
    )
    wide.to_csv(output_dir / "paired_holdout_rms.csv")

    comparisons = [
        ("one_fixed", "two_fixed"),
        ("two_fixed", "two_free"),
        ("two_fixed", "three_fixed"),
    ]
    summary = {}
    for simpler, complex_model in comparisons:
        if not {simpler, complex_model}.issubset(wide.columns):
            continue
        paired = wide.dropna(subset=[simpler, complex_model])
        if paired.empty:
            continue
        improvement = paired[simpler] - paired[complex_model]
        key = f"{simpler}_to_{complex_model}"
        summary[key] = {
            "n_matched_spectra": int(len(paired)),
            "fraction_new_model_better": float((improvement > 0).mean()),
            "median_paired_holdout_improvement_dex": float(
                improvement.median()
            ),
            "mean_paired_holdout_improvement_dex": float(
                improvement.mean()
            ),
        }

    # Retain original V3 paired-summary keys where possible.
    if set(("one_fixed", "two_fixed", "two_free")).issubset(wide.columns):
        paired = wide.dropna(
            subset=["one_fixed", "two_fixed", "two_free"]
        )
        if len(paired):
            summary.update({
                "n_matched_spectra": int(len(paired)),
                "fraction_two_fixed_better_than_one_fixed": float(
                    (paired["two_fixed"] < paired["one_fixed"]).mean()
                ),
                "fraction_two_free_better_than_two_fixed": float(
                    (paired["two_free"] < paired["two_fixed"]).mean()
                ),
                "median_paired_holdout_improvement_one_to_two_fixed_dex":
                    float(np.median(
                        paired["one_fixed"] - paired["two_fixed"]
                    )),
                "median_paired_holdout_improvement_two_fixed_to_free_dex":
                    float(np.median(
                        paired["two_fixed"] - paired["two_free"]
                    )),
            })

    (output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not MODELS:
        raise ValueError("MODELS cannot be empty.")
    if len(set(MODELS)) != len(MODELS):
        raise ValueError("MODELS contains duplicates.")
    for model in MODELS:
        if model not in VALID_MODELS:
            raise ValueError(f"Invalid model: {model}")
    if MAX_STARS < 0:
        raise ValueError("MAX_STARS must be >= 0.")
    if not 0 <= PLOT_FRACTION <= 1:
        raise ValueError("PLOT_FRACTION must be between 0 and 1.")
    if N_STARTS < 1:
        raise ValueError("N_STARTS must be >= 1.")
    if HOLDOUT_EVERY < 3:
        raise ValueError("HOLDOUT_EVERY must be >= 3.")
    if not 0 < MIN_FREQUENCY < MAX_FREQUENCY:
        raise ValueError("Invalid frequency limits.")

    results_file = OUTPUT_DIR / "fit_results.csv"
    if results_file.exists() and not ALLOW_OVERWRITE:
        raise FileExistsError(
            f"Results already exist:\n{results_file}\n\n"
            "Change OUTPUT_DIR or set ALLOW_OVERWRITE=True."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nu_max_lookup, valid_pairs = load_manifest(MANIFEST_PATH)
    files = []
    for path in sorted(INPUT_DIR.glob("*.npz")):
        try:
            kic, duration = parse_filename(path)
        except ValueError:
            continue
        if kic in nu_max_lookup and (kic, duration) in valid_pairs:
            files.append((path, kic, duration))

    if not files:
        raise ValueError("No physical PSDs match the manifest.")

    if MAX_STARS > 0:
        available = np.array(
            sorted({kic for _, kic, _ in files}), dtype=np.int64
        )
        rng = np.random.default_rng(SAMPLE_SEED)
        selected = set(
            rng.choice(
                available,
                size=min(MAX_STARS, len(available)),
                replace=False,
            ).tolist()
        )
        files = [item for item in files if item[1] in selected]

    plot_kics = (
        select_plot_kics(files, PLOT_FRACTION, PLOT_SEED)
        if MAKE_PLOTS else set()
    )
    expected_plot_count = sum(
        kic in plot_kics for _, kic, _ in files
    )

    config = {
        "models": list(MODELS),
        "max_stars": MAX_STARS,
        "sample_seed": SAMPLE_SEED,
        "n_starts": N_STARTS,
        "max_nfev": MAX_NFEV,
        "robust_loss": ROBUST_LOSS,
        "robust_f_scale": ROBUST_F_SCALE,
        "holdout_every": HOLDOUT_EVERY,
        "min_frequency_uHz": MIN_FREQUENCY,
        "max_frequency_uHz": MAX_FREQUENCY,
        "n_log_bins": N_LOG_BINS,
        "min_bin_points": MIN_BIN_POINTS,
        "min_raw_points": MIN_RAW_POINTS,
        "min_populated_bins": MIN_POPULATED_BINS,
        "fixed_alpha": FIXED_ALPHA,
        "free_alpha_bounds": [ALPHA_MIN, ALPHA_MAX],
        "envelope_width_bounds": [
            ENVELOPE_WIDTH_MIN, ENVELOPE_WIDTH_MAX
        ],
        "two_start_ratios": list(TWO_START_RATIOS),
        "three_start_patterns": [
            list(pattern) for pattern in THREE_START_PATTERNS
        ],
        "make_plots": MAKE_PLOTS,
        "plot_fraction": PLOT_FRACTION,
        "plot_seed": PLOT_SEED,
        "plot_kic_count": len(plot_kics),
        "expected_plot_count": expected_plot_count,
        "plot_kics": sorted(plot_kics),
        "plot_dir": str(PLOT_DIR),
        "manifest": str(MANIFEST_PATH),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "filter_transfer_function_applied": False,
        "kepler_integration_response_applied": False,
        "colored_noise_component_applied": False,
        "nu_char_nu_max_scaling_constraint_applied": False,
        "objective": "log10 median PSD, soft_l1, f_scale=0.15",
        "holdout": "every fifth populated log-frequency bin",
        "final_fit": "all bins, initialized from best training fit",
        "filter_duration_is_observation_length": False,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2)
    )

    rows, failures = [], []
    n_stars = len({kic for _, kic, _ in files})
    print(
        f"Fitting {len(files)} spectra from {n_stars} unique KICs.",
        flush=True,
    )
    print(f"Models: {MODELS}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    if MAKE_PLOTS:
        print(
            f"Diagnostic plots: {len(plot_kics)}/{n_stars} KICs "
            f"({PLOT_FRACTION:.1%}), up to {expected_plot_count} plots "
            f"in {PLOT_DIR}.",
            flush=True,
        )

    for i, (path, kic, duration) in enumerate(files, start=1):
        try:
            with np.load(path) as data:
                freq = np.asarray(data["frequency_uHz"], dtype=float)
                power = np.asarray(data["power"], dtype=float)

            good = (
                np.isfinite(freq)
                & np.isfinite(power)
                & (freq >= MIN_FREQUENCY)
                & (freq <= MAX_FREQUENCY)
                & (power > 0)
            )
            freq, power = freq[good], power[good]
            if len(freq) < MIN_RAW_POINTS:
                raise ValueError("Too few valid PSD samples.")

            order = np.argsort(freq)
            freq, power = freq[order], power[order]
            xb, yb, _ = log_bin(
                freq, power, bins=N_LOG_BINS,
                min_points=MIN_BIN_POINTS,
            )
            if len(xb) < MIN_POPULATED_BINS:
                raise ValueError("Too few populated log bins.")

            nu_max = nu_max_lookup[kic]
            if not xb.min() < nu_max < xb.max():
                raise ValueError(
                    f"nu_max={nu_max} outside binned PSD range "
                    f"[{xb.min()}, {xb.max()}]"
                )

            successes = {}
            for model in MODELS:
                try:
                    model_index = VALID_MODELS.index(model)
                    seed_sequence = np.random.SeedSequence([
                        SAMPLE_SEED, int(kic),
                        int(duration[:-1]), model_index,
                    ])
                    rng = np.random.default_rng(seed_sequence)
                    fit_info = fit_variant(
                        xb, yb, nu_max, model, rng
                    )
                    successes[model] = fit_info
                    rows.append(
                        result_row(
                            path, kic, duration, nu_max, xb,
                            freq, model, fit_info,
                        )
                    )
                except Exception as exc:
                    failures.append({
                        "file": path.name,
                        "kic": kic,
                        "duration": duration,
                        "model": model,
                        "error": repr(exc),
                    })

            if MAKE_PLOTS and kic in plot_kics and successes:
                try:
                    plot_fit(
                        path, kic, duration, nu_max, freq,
                        power, xb, yb, successes,
                        PLOT_DIR / f"{path.stem}.png",
                    )
                except Exception as exc:
                    print(
                        f"Plot warning for {path.name}: {exc}",
                        flush=True,
                    )
        except Exception as exc:
            for model in MODELS:
                failures.append({
                    "file": path.name,
                    "kic": kic,
                    "duration": duration,
                    "model": model,
                    "error": repr(exc),
                })

        if i % SAVE_EVERY == 0 or i == len(files):
            save_tables(rows, failures, OUTPUT_DIR)
            print(
                f"{i}/{len(files)} spectra | "
                f"successful fits={len(rows)} | "
                f"failures={len(failures)}",
                flush=True,
            )

    write_summary(rows, OUTPUT_DIR)
    print("\nFinished.", flush=True)
    print(f"Successful fits: {len(rows)}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    print(f"Results: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
