
#!/usr/bin/env python3
"""
Final V3 granulation fitting for physical Kepler PSDs.

Primary model:
    two_fixed : Two Harvey components, alpha = 4

Optional comparison models:
    one_fixed : One Harvey component, alpha = 4
    two_free  : Two Harvey components, alpha in [2, 6]

The complete PSD model contains:
    - One or two Harvey components
    - A constant white-noise floor
    - A Gaussian oscillation envelope centred at catalogue nu_max

Fitting:
    - Logarithmic frequency bins
    - Median PSD power in each bin
    - Robust soft-L1 least squares in log10 power
    - Multiple initializations
    - Interleaved held-out bins for diagnostic comparison
    - Final refitting using all bins

The held-out bins are not statistically independent of the
training bins. Their residuals are diagnostic, not formal
model-selection statistics.

All configuration is defined below.
No command-line arguments are required.

Run:
    python fit_granulation_v3_final.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "Baseline" / "manifest.csv"

INPUT_DIR = PROJECT_ROOT / "data" / "psd_physical"


# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

# FINAL PRIMARY MODEL:
MODELS = ("two_free",)

# For comparison runs, change to:
# MODELS = ("one_fixed", "two_fixed", "two_free")

# Or:
# MODELS = ("two_free",)


# ------------------------------------------------------------
# Sample selection
# ------------------------------------------------------------

# 0 = full catalogue
# 300 = 300 randomly selected unique KICs
MAX_STARS = 0

SAMPLE_SEED = 42


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

N_STARTS = 4

MAX_NFEV = 12000

ROBUST_LOSS = "soft_l1"

ROBUST_F_SCALE = 0.15

HOLDOUT_EVERY = 5


# ------------------------------------------------------------
# PSD preprocessing
# ------------------------------------------------------------

MIN_FREQUENCY = 0.1       # microHz
MAX_FREQUENCY = 283.0     # microHz

N_LOG_BINS = 180

MIN_BIN_POINTS = 4

MIN_RAW_POINTS = 100

MIN_POPULATED_BINS = 30


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

# Unique directory for the final full-catalogue fit.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "fitted_parameters_v3_two_free_all"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "results"
    / "granulation_v3_two_free_all_5pct_plots"
)

# Save plots for a reproducible random subset of fitted KICs.
MAKE_PLOTS = True

PLOT_FRACTION = 0.05

PLOT_SEED = 42

SAVE_EVERY = 25

# Protect existing results from accidental overwriting.
ALLOW_OVERWRITE = False


# ------------------------------------------------------------
# Numerical constants
# ------------------------------------------------------------

TINY = np.finfo(float).tiny

ALPHA_MIN = 2.0
ALPHA_MAX = 6.0

FIXED_ALPHA = 4.0

ENVELOPE_WIDTH_MIN = 0.08
ENVELOPE_WIDTH_MAX = 0.8

VALID_MODELS = ("one_fixed", "two_fixed", "two_free")


# ============================================================
# FILE AND MANIFEST UTILITIES
# ============================================================

def parse_filename(path: Path):
    """
    Extract KIC and filtering duration from a PSD filename.
    """

    match = re.search(
        r"kic0*(\d+).*?(20|55|80)d",
        path.stem,
        flags=re.I,
    )

    if not match:
        raise ValueError(
            f"Cannot identify KIC/duration from {path.name}"
        )

    kic = int(match.group(1))
    duration = f"{match.group(2)}d"

    return kic, duration


def select_plot_kics(files, fraction, seed):
    """Choose a reproducible KIC sample for diagnostic plots."""
    available = np.array(
        sorted({kic for _, kic, _ in files}),
        dtype=np.int64,
    )
    if fraction <= 0 or not len(available):
        return set()

    sample_size = min(
        len(available),
        max(1, int(round(len(available) * fraction))),
    )
    rng = np.random.default_rng(seed)
    return set(
        rng.choice(
            available,
            size=sample_size,
            replace=False,
        ).tolist()
    )


def load_manifest(path: Path, tolerance=0.01):
    """
    Load catalogue nu_max values and valid KIC/duration pairs.
    """

    df = pd.read_csv(
        path,
        usecols=["kic", "duration", "nu_max"],
    )

    df["kic"] = pd.to_numeric(
        df["kic"],
        errors="raise",
    ).astype("int64")

    df["nu_max"] = pd.to_numeric(
        df["nu_max"],
        errors="coerce",
    )

    df["duration"] = (
        df["duration"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        np.isfinite(df["nu_max"])
        & (df["nu_max"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No positive finite nu_max values in manifest."
        )

    grouped = df.groupby("kic")["nu_max"]

    inconsistent = grouped.agg(
        lambda x: (
            (x.max() - x.min()) / x.median()
            > tolerance
        )
    )

    if inconsistent.any():
        bad = inconsistent[
            inconsistent
        ].index.tolist()[:15]

        raise ValueError(
            f"Inconsistent catalogue nu_max: {bad}"
        )

    nu_max_lookup = grouped.median().to_dict()

    valid_pairs = set(
        zip(df["kic"], df["duration"])
    )

    return nu_max_lookup, valid_pairs


# ============================================================
# PSD BINNING
# ============================================================

def log_bin(freq, power, bins=180, min_points=4):
    """
    Logarithmically bin the PSD.

    Frequency:
        Geometric mean within each populated bin.

    Power:
        Median within each populated bin.
    """

    edges = np.geomspace(
        freq.min(),
        freq.max(),
        bins + 1,
    )

    indices = (
        np.searchsorted(
            edges,
            freq,
            side="right",
        ) - 1
    )

    indices = np.clip(
        indices,
        0,
        bins - 1,
    )

    xs = []
    ys = []
    counts = []

    for j in range(bins):

        mask = indices == j

        count = int(mask.sum())

        if count < min_points:
            continue

        xs.append(
            float(
                np.exp(
                    np.mean(
                        np.log(freq[mask])
                    )
                )
            )
        )

        ys.append(
            float(
                np.median(power[mask])
            )
        )

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
    """
    Harvey-like background component.
    """

    return height / (
        1.0 + (nu / nu_char) ** alpha
    )


def integrated_rms(height, nu_char, alpha):
    """
    Integrated Harvey amplitude proxy.

    Integral from zero to infinity of:

        H / (1 + (nu / nu_char)^alpha)

    The result is expressed in native PSD amplitude units.

    This is a model-dependent extrapolated quantity.
    """

    integral = (
        height
        * nu_char
        * np.pi
        / (
            alpha
            * np.sin(np.pi / alpha)
        )
    )

    return float(np.sqrt(integral))


def unpack(theta, model):
    """
    Convert optimization parameters to physical parameters.

    Positive parameters are optimized in natural-log space.

    Components are sorted by characteristic frequency.
    """

    if model == "one_fixed":

        h, c, w, g, lw = theta

        components = [
            (
                float(np.exp(h)),
                float(np.exp(c)),
                FIXED_ALPHA,
            )
        ]

    elif model == "two_fixed":

        h1, c1, h2, c2, w, g, lw = theta

        components = [
            (
                float(np.exp(h1)),
                float(np.exp(c1)),
                FIXED_ALPHA,
            ),
            (
                float(np.exp(h2)),
                float(np.exp(c2)),
                FIXED_ALPHA,
            ),
        ]

    elif model == "two_free":

        (
            h1, c1, a1,
            h2, c2, a2,
            w, g, lw,
        ) = theta

        components = [
            (
                float(np.exp(h1)),
                float(np.exp(c1)),
                float(a1),
            ),
            (
                float(np.exp(h2)),
                float(np.exp(c2)),
                float(a2),
            ),
        ]

    else:
        raise ValueError(
            f"Unknown model: {model}"
        )

    components.sort(
        key=lambda component: component[1]
    )

    white = float(np.exp(w))
    amplitude = float(np.exp(g))
    width = float(np.exp(lw))

    return components, white, amplitude, width


def evaluate(theta, nu, nu_max, model):
    """
    Evaluate complete PSD model and individual components.
    """

    components, white, amplitude, width = unpack(
        theta,
        model,
    )

    parts = [
        harvey(nu, *component)
        for component in components
    ]

    white_part = np.full_like(
        nu,
        white,
        dtype=float,
    )

    envelope_part = amplitude * np.exp(
        -0.5 * (
            np.log(nu / nu_max) / width
        ) ** 2
    )

    parts.append(white_part)
    parts.append(envelope_part)

    total = np.sum(parts, axis=0)

    return total, parts


# ============================================================
# PARAMETER BOUNDS
# ============================================================

def make_bounds(xb, yb, model):
    """
    Construct bounds for the optimization parameters.

    No nu_max-dependent characteristic-frequency constraints
    are imposed.
    """

    fmin = float(xb.min())
    fmax = float(xb.max())

    ymax = max(
        float(np.percentile(yb, 90)),
        TINY * 1e20,
    )

    ymin = max(
        float(np.percentile(yb, 10) * 0.1),
        TINY * 1e20,
    )

    log_h_lo = np.log(ymin * 1e-3)
    log_h_hi = np.log(ymax * 1e3)

    log_c_lo = np.log(fmin / 10)
    log_c_hi = np.log(fmax * 10)

    tail_lo = [
        np.log(ymin * 1e-4),
        np.log(ymin * 1e-3),
        np.log(ENVELOPE_WIDTH_MIN),
    ]

    tail_hi = [
        np.log(ymax * 10),
        np.log(ymax * 1e3),
        np.log(ENVELOPE_WIDTH_MAX),
    ]

    if model == "one_fixed":

        lower = [
            log_h_lo,
            log_c_lo,
            *tail_lo,
        ]

        upper = [
            log_h_hi,
            log_c_hi,
            *tail_hi,
        ]

    elif model == "two_fixed":

        lower = [
            log_h_lo,
            log_c_lo,
            log_h_lo,
            log_c_lo,
            *tail_lo,
        ]

        upper = [
            log_h_hi,
            log_c_hi,
            log_h_hi,
            log_c_hi,
            *tail_hi,
        ]

    elif model == "two_free":

        lower = [
            log_h_lo,
            log_c_lo,
            ALPHA_MIN,
            log_h_lo,
            log_c_lo,
            ALPHA_MIN,
            *tail_lo,
        ]

        upper = [
            log_h_hi,
            log_c_hi,
            ALPHA_MAX,
            log_h_hi,
            log_c_hi,
            ALPHA_MAX,
            *tail_hi,
        ]

    else:
        raise ValueError(model)

    return (
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


# ============================================================
# INITIAL PARAMETERS
# ============================================================

def make_initial(
    xb,
    yb,
    nu_max,
    model,
    start,
    rng,
):
    """
    Construct one initialization.

    The first initialization is deterministic.

    Subsequent initializations are randomized.
    """

    fmin = float(xb.min())
    fmax = float(xb.max())

    ymax = float(
        np.percentile(yb, 90)
    )

    ymin = max(
        float(np.percentile(yb, 10) * 0.1),
        TINY * 1e20,
    )

    if model == "one_fixed":

        frequencies = [
            nu_max / 3
        ]

        heights = [
            max(ymax * 0.8, ymin)
        ]

    else:

        frequencies = [
            nu_max / 20,
            nu_max / 3,
        ]

        heights = [
            max(ymax * 0.7, ymin),
            max(ymax * 0.3, ymin),
        ]

    if start > 0:

        frequencies = [
            f * np.exp(
                rng.normal(0, 1.0)
            )
            for f in frequencies
        ]

        heights = [
            h * np.exp(
                rng.normal(0, 1.0)
            )
            for h in heights
        ]

    frequencies = [
        float(
            np.clip(
                f,
                fmin * 1.1,
                fmax / 1.1,
            )
        )
        for f in frequencies
    ]

    parameters = []

    for height, frequency in zip(
        heights,
        frequencies,
    ):

        parameters.extend([
            np.log(height),
            np.log(frequency),
        ])

        if model == "two_free":

            alpha = (
                FIXED_ALPHA
                if start == 0
                else float(
                    rng.uniform(2.2, 5.8)
                )
            )

            parameters.append(alpha)

    white = (
        ymin
        if start == 0
        else ymin * np.exp(
            rng.normal(0, 0.5)
        )
    )

    envelope = max(
        ymax * 0.3,
        ymin,
    )

    if start > 0:
        envelope *= np.exp(
            rng.normal(0, 0.7)
        )

    width = (
        0.35
        if start == 0
        else float(
            rng.uniform(0.10, 0.65)
        )
    )

    parameters.extend([
        np.log(white),
        np.log(envelope),
        np.log(width),
    ])

    return np.asarray(
        parameters,
        dtype=float,
    )


# ============================================================
# FITTING
# ============================================================

def log_residual(
    theta,
    x,
    y,
    nu_max,
    model,
):
    """
    Residuals in log10 PSD power.
    """

    prediction, _ = evaluate(
        theta,
        x,
        nu_max,
        model,
    )

    return (
        np.log10(
            np.maximum(prediction, TINY)
        )
        - np.log10(y)
    )


def fit_variant(
    xb,
    yb,
    nu_max,
    model,
    rng,
):
    """
    Fit one model to one PSD.

    Procedure:

    1. Create train/holdout frequency bins.
    2. Optimize from multiple initializations.
    3. Select the converged training fit with lowest cost.
    4. Evaluate train and held-out residuals.
    5. Refit all bins for final parameters.
    """

    holdout = (
        np.arange(len(xb))
        % HOLDOUT_EVERY
        == 0
    )

    training = ~holdout

    if (
        training.sum() < 25
        or holdout.sum() < 5
    ):
        raise ValueError(
            "Too few train/holdout bins."
        )

    lower, upper = make_bounds(
        xb,
        yb,
        model,
    )

    candidates = []

    for start in range(N_STARTS):

        p0 = make_initial(
            xb,
            yb,
            nu_max,
            model,
            start,
            rng,
        )

        p0 = np.clip(
            p0,
            lower + 1e-8,
            upper - 1e-8,
        )

        try:

            fit = least_squares(
                log_residual,
                p0,
                args=(
                    xb[training],
                    yb[training],
                    nu_max,
                    model,
                ),
                bounds=(lower, upper),
                loss=ROBUST_LOSS,
                f_scale=ROBUST_F_SCALE,
                max_nfev=MAX_NFEV,
            )

            # Only converged, finite solutions are accepted.
            if (
                fit.success
                and np.all(np.isfinite(fit.x))
                and np.isfinite(fit.cost)
            ):

                candidates.append(
                    (
                        float(fit.cost),
                        start,
                        fit,
                    )
                )

        except (
            FloatingPointError,
            ValueError,
            OverflowError,
        ):
            continue

    if not candidates:

        raise RuntimeError(
            "No converged training fit "
            "from any initialization."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    _, best_start, best_train = candidates[0]

    train_residuals = log_residual(
        best_train.x,
        xb[training],
        yb[training],
        nu_max,
        model,
    )

    holdout_residuals = log_residual(
        best_train.x,
        xb[holdout],
        yb[holdout],
        nu_max,
        model,
    )

    # Refit all bins from the selected solution.
    final = least_squares(
        log_residual,
        best_train.x,
        args=(
            xb,
            yb,
            nu_max,
            model,
        ),
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
        max_nfev=MAX_NFEV,
    )

    if not final.success:

        raise RuntimeError(
            "Full-data optimizer did not converge: "
            + str(final.message)
        )

    if (
        not np.all(np.isfinite(final.x))
        or not np.isfinite(final.cost)
    ):

        raise RuntimeError(
            "Nonfinite full-data fit."
        )

    full_residuals = log_residual(
        final.x,
        xb,
        yb,
        nu_max,
        model,
    )

    return {
        "fit": final,

        "n_successful_starts": len(candidates),

        "best_start": best_start,

        "train_rms_log10": float(
            np.sqrt(
                np.mean(train_residuals ** 2)
            )
        ),

        "holdout_rms_log10": float(
            np.sqrt(
                np.mean(holdout_residuals ** 2)
            )
        ),

        "full_rms_log10": float(
            np.sqrt(
                np.mean(full_residuals ** 2)
            )
        ),

        "n_train_bins": int(training.sum()),

        "n_holdout_bins": int(holdout.sum()),

        "train_success": True,

        "full_success": True,

        "train_message": str(
            best_train.message
        ),

        "full_message": str(
            final.message
        ),
    }


# ============================================================
# OUTPUT ROW
# ============================================================

def result_row(
    path,
    kic,
    duration,
    nu_max,
    xb,
    freq,
    model,
    fit_info,
):
    """
    Convert one fit into a CSV result row.

    Column names are compatible with the V3 pilot
    and the V3 stability-analysis script.
    """

    fit = fit_info["fit"]

    components, white, amplitude, width = unpack(
        fit.x,
        model,
    )

    row = {
        "file": path.name,
        "kic": kic,
        "duration": duration,
        "nu_max_catalogue": nu_max,
        "model": model,

        "success": True,

        "cost": float(fit.cost),

        "rms_log10_residual":
            fit_info["full_rms_log10"],

        "train_rms_log10_residual":
            fit_info["train_rms_log10"],

        "holdout_rms_log10_residual":
            fit_info["holdout_rms_log10"],

        "n_successful_starts":
            fit_info["n_successful_starts"],

        "best_start":
            fit_info["best_start"],

        "n_raw_points": len(freq),

        "n_fit_bins": len(xb),

        "n_train_bins":
            fit_info["n_train_bins"],

        "n_holdout_bins":
            fit_info["n_holdout_bins"],

        "white_noise": white,

        "oscillation_envelope_amplitude":
            amplitude,

        "oscillation_envelope_centre_uHz":
            nu_max,

        "oscillation_envelope_log_width":
            width,

        "message":
            fit_info["full_message"],

        "long_component_height": np.nan,
        "long_component_nu_char_uHz": np.nan,
        "long_component_tau_days": np.nan,
        "long_component_alpha": np.nan,
        "long_integrated_rms_native_sqrt_uHz": np.nan,

        "short_component_height": np.nan,
        "short_component_nu_char_uHz": np.nan,
        "short_component_tau_days": np.nan,
        "short_component_alpha": np.nan,
        "short_integrated_rms_native_sqrt_uHz": np.nan,

        "long_alpha_at_bound": False,
        "short_alpha_at_bound": False,

        "long_char_below_fit_band": False,
        "short_char_below_fit_band": False,

        "long_char_below_filter_timescale": False,
        "short_char_below_filter_timescale": False,
    }

    if len(components) == 2:
        labels = ["long", "short"]
    else:
        labels = ["short"]

    # In one_fixed, the only Harvey component is stored
    # in the short columns for compatibility.
    # It is NOT necessarily the physical short component.

    for label, (height, char, alpha) in zip(
        labels,
        components,
    ):

        row[
            f"{label}_component_height"
        ] = height

        row[
            f"{label}_component_nu_char_uHz"
        ] = char

        row[
            f"{label}_component_tau_days"
        ] = (
            1.0
            / (
                2
                * np.pi
                * char
                * 1e-6
                * 86400
            )
        )

        row[
            f"{label}_component_alpha"
        ] = alpha

        row[
            f"{label}_integrated_rms_native_sqrt_uHz"
        ] = integrated_rms(
            height,
            char,
            alpha,
        )

        row[
            f"{label}_alpha_at_bound"
        ] = bool(
            model == "two_free"
            and min(
                abs(alpha - ALPHA_MIN),
                abs(alpha - ALPHA_MAX),
            ) <= 0.01
        )

        row[
            f"{label}_char_below_fit_band"
        ] = bool(
            char < float(xb.min())
        )

        # 1/T is only a heuristic frequency scale.
        # It is not a calibrated KADACS transfer function.

        filter_frequency = (
            1e6
            / (
                int(duration[:-1])
                * 86400
            )
        )

        row[
            f"{label}_char_below_filter_timescale"
        ] = bool(
            char < filter_frequency
        )

    return row


# ============================================================
# PLOTTING
# ============================================================

def plot_fit(
    path,
    kic,
    duration,
    nu_max,
    freq,
    power,
    xb,
    yb,
    results,
    output,
):
    """
    Plot:
        1. Full PSD and model comparisons.
        2. Selected-model decomposition.
        3. Residuals.
        4. Train/holdout residual comparison.

    The lower panels use independent x axes to avoid
    the shared-axis conflict in the original pilot script.
    """

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 12),
        sharex=False,
        gridspec_kw={
            "height_ratios": [3, 1, 1]
        },
    )

    dense = np.geomspace(
        freq.min(),
        freq.max(),
        1000,
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.loglog(
        freq,
        power,
        ".",
        alpha=0.10,
        ms=1,
        color="0.5",
        label="Raw PSD",
    )

    ax.loglog(
        xb,
        yb,
        ".",
        ms=4,
        color="black",
        label="Binned median",
    )

    for model in MODELS:

        if model not in results:
            continue

        prediction, _ = evaluate(
            results[model]["fit"].x,
            dense,
            nu_max,
            model,
        )

        ax.loglog(
            dense,
            prediction,
            lw=1.7,
            label=model,
        )

    ax.axvline(
        nu_max,
        color="0.4",
        ls=":",
        label=r"Catalogue $\nu_{\max}$",
    )

    ax.set(
        title="Model comparison",
        xlabel="Frequency (microHz)",
        ylabel="PSD (native units)",
    )

    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    # --------------------------------------------------------
    # Component decomposition
    # --------------------------------------------------------

    ax2 = axes[0, 1]

    if "two_fixed" in results:
        chosen = "two_fixed"
    else:
        chosen = next(iter(results))

    prediction, parts = evaluate(
        results[chosen]["fit"].x,
        dense,
        nu_max,
        chosen,
    )

    ax2.loglog(
        xb,
        yb,
        ".",
        ms=4,
        color="black",
        label="Binned median",
    )

    ax2.loglog(
        dense,
        prediction,
        lw=2,
        label=chosen,
    )

    if chosen == "one_fixed":
        names = ["Harvey"]
    else:
        names = [
            "Long Harvey",
            "Short Harvey",
        ]

    names += [
        "White noise",
        "Oscillation envelope",
    ]

    for component, name in zip(parts, names):

        ax2.loglog(
            dense,
            component,
            "--",
            alpha=0.75,
            label=name,
        )

    ax2.axvline(
        nu_max,
        color="0.4",
        ls=":",
    )

    ax2.set(
        title=f"{chosen} decomposition",
        xlabel="Frequency (microHz)",
        ylabel="PSD (native units)",
    )

    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_ylim(ax.get_ylim())

    # --------------------------------------------------------
    # Residual panels
    # --------------------------------------------------------

    for col in range(2):

        residual_ax = axes[1, col]

        for model in MODELS:

            if model not in results:
                continue

            info = results[model]

            residual = log_residual(
                info["fit"].x,
                xb,
                yb,
                nu_max,
                model,
            )

            residual_ax.semilogx(
                xb,
                -residual,
                ".",
                ms=3,
                label=model,
            )

        residual_ax.axhline(
            0,
            color="black",
            lw=1,
        )

        residual_ax.set(
            xlabel="Frequency (microHz)",
            ylabel="log10(data/fit)",
        )

        residual_ax.grid(
            True,
            which="both",
            alpha=0.2,
        )

    # --------------------------------------------------------
    # Train/holdout comparison
    # --------------------------------------------------------

    for col in range(2):

        metric_ax = axes[2, col]

        for model in MODELS:

            if model not in results:
                continue

            info = results[model]

            metric_ax.plot(
                [0, 1],
                [
                    info["train_rms_log10"],
                    info["holdout_rms_log10"],
                ],
                "o-",
                label=model,
            )

        metric_ax.set(
            xticks=[0, 1],
            xticklabels=[
                "Train",
                "Held out",
            ],
            ylabel="log10 RMS (dex)",
        )

        metric_ax.grid(
            True,
            alpha=0.2,
        )

    axes[2, 0].legend(fontsize=8)

    fig.suptitle(
        f"KIC {kic} | {duration} | "
        f"$\\nu_{{\\max}}$={nu_max:.2f} microHz"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=130,
    )

    plt.close(fig)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_tables(rows, failures, output_dir):
    """
    Save fit results and failures.

    Files are written to temporary paths first and then
    replaced, reducing the risk of partially written CSVs.
    """

    results_path = output_dir / "fit_results.csv"
    failures_path = output_dir / "fit_failures.csv"

    results_temp = output_dir / "fit_results.tmp"
    failures_temp = output_dir / "fit_failures.tmp"

    pd.DataFrame(rows).to_csv(
        results_temp,
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=[
            "file",
            "kic",
            "duration",
            "model",
            "error",
        ],
    ).to_csv(
        failures_temp,
        index=False,
    )

    results_temp.replace(results_path)
    failures_temp.replace(failures_path)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def write_summary(rows, output_dir):
    """
    Write model summaries and paired holdout comparisons.
    """

    df = pd.DataFrame(rows)

    if df.empty:

        (
            output_dir / "model_comparison.csv"
        ).write_text(
            "model,n,median_full_rms,"
            "median_holdout_rms\n"
        )

        return

    statistics = []

    for model, part in df.groupby("model"):

        statistics.append({
            "model": model,

            "n": len(part),

            "median_full_rms":
                part["rms_log10_residual"].median(),

            "median_train_rms":
                part["train_rms_log10_residual"].median(),

            "median_holdout_rms":
                part["holdout_rms_log10_residual"].median(),

            "long_char_below_fit_band_fraction":
                part["long_char_below_fit_band"].mean(),

            "long_char_below_filter_timescale_fraction":
                part[
                    "long_char_below_filter_timescale"
                ].mean(),

            "long_alpha_at_bound_fraction":
                part["long_alpha_at_bound"].mean(),
        })

    pd.DataFrame(statistics).to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    wide = df.pivot_table(
        index=["kic", "duration"],
        columns="model",
        values="holdout_rms_log10_residual",
        aggfunc="first",
    )

    wide.to_csv(
        output_dir / "paired_holdout_rms.csv"
    )

    if set(VALID_MODELS).issubset(wide.columns):

        paired = wide.dropna(
            subset=list(VALID_MODELS)
        )

        if len(paired):

            summary = {
                "n_matched_spectra":
                    len(paired),

                "fraction_two_fixed_better_than_one_fixed":
                    float(
                        (
                            paired["two_fixed"]
                            < paired["one_fixed"]
                        ).mean()
                    ),

                "fraction_two_free_better_than_two_fixed":
                    float(
                        (
                            paired["two_free"]
                            < paired["two_fixed"]
                        ).mean()
                    ),

                "median_paired_holdout_improvement_one_to_two_fixed_dex":
                    float(
                        np.median(
                            paired["one_fixed"]
                            - paired["two_fixed"]
                        )
                    ),

                "median_paired_holdout_improvement_two_fixed_to_free_dex":
                    float(
                        np.median(
                            paired["two_fixed"]
                            - paired["two_free"]
                        )
                    ),
            }

            (
                output_dir / "paired_summary.json"
            ).write_text(
                json.dumps(
                    summary,
                    indent=2,
                )
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if not MODELS:
        raise ValueError(
            "MODELS cannot be empty."
        )

    if len(set(MODELS)) != len(MODELS):
        raise ValueError(
            "MODELS contains duplicates."
        )

    for model in MODELS:

        if model not in VALID_MODELS:
            raise ValueError(
                f"Invalid model: {model}"
            )

    if MAX_STARS < 0:
        raise ValueError(
            "MAX_STARS must be >= 0."
        )

    if not 0 <= PLOT_FRACTION <= 1:
        raise ValueError(
            "PLOT_FRACTION must be between 0 and 1."
        )

    if N_STARTS < 1:
        raise ValueError(
            "N_STARTS must be >= 1."
        )

    if HOLDOUT_EVERY < 3:
        raise ValueError(
            "HOLDOUT_EVERY must be >= 3."
        )

    if not (
        0 < MIN_FREQUENCY < MAX_FREQUENCY
    ):
        raise ValueError(
            "Invalid frequency limits."
        )

    # --------------------------------------------------------
    # Protect previous results
    # --------------------------------------------------------

    results_file = (
        OUTPUT_DIR / "fit_results.csv"
    )

    if (
        results_file.exists()
        and not ALLOW_OVERWRITE
    ):

        raise FileExistsError(
            f"Results already exist:\n"
            f"{results_file}\n\n"
            "Change OUTPUT_DIR or set "
            "ALLOW_OVERWRITE=True."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load catalogue
    # --------------------------------------------------------

    nu_max_lookup, valid_pairs = load_manifest(
        MANIFEST_PATH
    )

    files = []

    for path in sorted(
        INPUT_DIR.glob("*.npz")
    ):

        try:
            kic, duration = parse_filename(path)
        except ValueError:
            continue

        if (
            kic in nu_max_lookup
            and (kic, duration) in valid_pairs
        ):

            files.append(
                (
                    path,
                    kic,
                    duration,
                )
            )

    if not files:

        raise ValueError(
            "No physical PSDs match the manifest."
        )

    # --------------------------------------------------------
    # Select unique stars
    # --------------------------------------------------------

    if MAX_STARS > 0:

        available = np.array(
            sorted({
                kic
                for _, kic, _ in files
            }),
            dtype=np.int64,
        )

        rng = np.random.default_rng(
            SAMPLE_SEED
        )

        selected = set(
            rng.choice(
                available,
                size=min(
                    MAX_STARS,
                    len(available),
                ),
                replace=False,
            ).tolist()
        )

        files = [
            item
            for item in files
            if item[1] in selected
        ]

    # Sample KICs, rather than individual spectra, so plotted examples retain
    # their available duration series while keeping the plot count near 5%.
    plot_kics = (
        select_plot_kics(files, PLOT_FRACTION, PLOT_SEED)
        if MAKE_PLOTS
        else set()
    )

    expected_plot_count = sum(
        kic in plot_kics
        for _, kic, _ in files
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

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

        "fixed_alpha": FIXED_ALPHA,

        "free_alpha_bounds": [
            ALPHA_MIN,
            ALPHA_MAX,
        ],

        "envelope_width_bounds": [
            ENVELOPE_WIDTH_MIN,
            ENVELOPE_WIDTH_MAX,
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

        "nu_char_nu_max_scaling_constraint_applied": False,

        "objective":
            "log10 median PSD, soft_l1, f_scale=0.15",

        "holdout":
            "every fifth populated log-frequency bin",

        "final_fit":
            "all bins, initialized from best training fit",
    }

    (
        OUTPUT_DIR / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Fit spectra
    # --------------------------------------------------------

    rows = []
    failures = []

    n_stars = len({
        kic
        for _, kic, _ in files
    })

    print(
        f"Fitting {len(files)} spectra "
        f"from {n_stars} unique KICs.",
        flush=True,
    )

    print(
        f"Models: {MODELS}",
        flush=True,
    )

    print(
        f"Output: {OUTPUT_DIR}",
        flush=True,
    )

    if MAKE_PLOTS:
        print(
            f"Diagnostic plots: {len(plot_kics)} / {n_stars} KICs "
            f"({PLOT_FRACTION:.1%}), up to {expected_plot_count} plots "
            f"in {PLOT_DIR}.",
            flush=True,
        )

    for i, (path, kic, duration) in enumerate(
        files,
        start=1,
    ):

        try:

            with np.load(path) as data:

                freq = np.asarray(
                    data["frequency_uHz"],
                    dtype=float,
                )

                power = np.asarray(
                    data["power"],
                    dtype=float,
                )

            good = (
                np.isfinite(freq)
                & np.isfinite(power)
                & (freq >= MIN_FREQUENCY)
                & (freq <= MAX_FREQUENCY)
                & (power > 0)
            )

            freq = freq[good]
            power = power[good]

            if len(freq) < MIN_RAW_POINTS:

                raise ValueError(
                    "Too few valid PSD samples."
                )

            order = np.argsort(freq)

            freq = freq[order]
            power = power[order]

            xb, yb, _ = log_bin(
                freq,
                power,
                bins=N_LOG_BINS,
                min_points=MIN_BIN_POINTS,
            )

            if len(xb) < MIN_POPULATED_BINS:

                raise ValueError(
                    "Too few populated log bins."
                )

            nu_max = nu_max_lookup[kic]

            if not (
                xb.min() < nu_max < xb.max()
            ):

                raise ValueError(
                    f"nu_max={nu_max} outside "
                    f"binned PSD range "
                    f"[{xb.min()}, {xb.max()}]"
                )

            successes = {}

            for model in MODELS:

                try:

                    model_index = VALID_MODELS.index(
                        model
                    )

                    seed_sequence = np.random.SeedSequence([
                        SAMPLE_SEED,
                        int(kic),
                        int(duration[:-1]),
                        model_index,
                    ])

                    rng = np.random.default_rng(
                        seed_sequence
                    )

                    fit_info = fit_variant(
                        xb,
                        yb,
                        nu_max,
                        model,
                        rng,
                    )

                    successes[model] = fit_info

                    rows.append(
                        result_row(
                            path,
                            kic,
                            duration,
                            nu_max,
                            xb,
                            freq,
                            model,
                            fit_info,
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

            # ------------------------------------------------
            # Optional plots
            # ------------------------------------------------

            if MAKE_PLOTS and kic in plot_kics and successes:

                try:

                    plot_fit(
                        path,
                        kic,
                        duration,
                        nu_max,
                        freq,
                        power,
                        xb,
                        yb,
                        successes,
                        PLOT_DIR / f"{path.stem}.png",
                    )

                except Exception as exc:

                    print(
                        f"Plot warning for "
                        f"{path.name}: {exc}",
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

        # ----------------------------------------------------
        # Periodic saving
        # ----------------------------------------------------

        if (
            i % SAVE_EVERY == 0
            or i == len(files)
        ):

            save_tables(
                rows,
                failures,
                OUTPUT_DIR,
            )

            print(
                f"{i}/{len(files)} spectra | "
                f"successful fits={len(rows)} | "
                f"failures={len(failures)}",
                flush=True,
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    write_summary(
        rows,
        OUTPUT_DIR,
    )

    print(
        "\nFinished.",
        flush=True,
    )

    print(
        f"Successful fits: {len(rows)}",
        flush=True,
    )

    print(
        f"Failures: {len(failures)}",
        flush=True,
    )

    print(
        f"Results: {OUTPUT_DIR.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
