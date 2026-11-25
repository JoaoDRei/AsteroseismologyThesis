
#!/usr/bin/env python3

"""
Analyse interpretation runs and compare Integrated Gradients
with SmoothGrad.

Outputs:
    interpretation_runs.csv
    interpretation_target_coverage.csv
    interpretation_agreement_per_star.csv
    interpretation_agreement_summary.csv

Also generates:
    - Copied saliency maps
    - Paired attribution overlays
    - Attribution scatter plots
    - Agreement heatmaps
    - Frequency-localisation heatmaps
    - Per-star metric boxplots

Run:
    python -m Baseline.analysis.analyze_interpretation
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Baseline.paths import BASELINE_DIR


ROOT = BASELINE_DIR.parent

DEFAULT_PLAN = (
    BASELINE_DIR
    / "interpretation"
    / "interpretation_plan.csv"
)

DEFAULT_STATUS_DIR = (
    BASELINE_DIR
    / "interpretation"
    / "status"
)

DEFAULT_OUTPUT_DIR = (
    BASELINE_DIR
    / "results_analysis"
)

AGREEMENT_DIR_NAME = "interpretation_agreement"


# ============================================================
# General utilities
# ============================================================

def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def resolve_path(value):
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (ROOT / path).resolve()


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan

    except (TypeError, ValueError):
        return np.nan


def targets_from_row(row):
    return [
        target.strip()
        for target in (row.get("targets") or "").split("+")
        if target.strip()
    ]


def status_output_dir(status, run_dir):
    value = status.get("output_dir")

    if value:
        output_dir = Path(value).expanduser()

        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir

        output_dir = output_dir.resolve()

        if output_dir.is_dir():
            return output_dir

    return run_dir / "interpretability"


def sample_key(path, target, method):
    """
    Extract sample ID from:
        kic_<sample_id>_<target>_<method>.npz/.png

    The sample ID may include duration.
    """
    stem = path.stem
    prefix = "kic_"
    suffix = f"_{target}_{method}"

    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unexpected attribution filename: {path}")

    return stem[len(prefix):-len(suffix)]


# ============================================================
# Collect run information
# ============================================================

def collect_run_record(row, status_dir, interpretation_dir):
    analysis_id = row["analysis_id"]

    status = read_json(
        status_dir / f"{analysis_id}.json"
    )

    run_dir = resolve_path(row["run_dir"])

    output_dir = status_output_dir(
        status,
        run_dir,
    )

    method = (
        row.get("method") or ""
    ).strip().lower()

    targets = targets_from_row(row)

    map_paths = []
    sample_ids = set()
    kics = set()
    target_records = []

    for target in targets:
        suffix = f"_{target}_{method}.png"

        target_paths = (
            sorted(
                path
                for path in output_dir.glob(f"kic_*{suffix}")
                if path.is_file()
            )
            if output_dir.is_dir()
            else []
        )

        target_output_dir = (
            interpretation_dir
            / analysis_id
            / target
        )

        if target_paths:
            target_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for path in target_paths:
                shutil.copy2(
                    path,
                    target_output_dir / path.name,
                )

        target_sample_ids = {
            sample_key(path, target, method)
            for path in target_paths
        }

        # KIC-only count is not reliable from filenames when
        # duration is included, so read metadata if available.
        target_kics = set()

        for path in target_paths:
            npz_path = path.with_suffix(".npz")

            if npz_path.is_file():
                try:
                    with np.load(
                        npz_path,
                        allow_pickle=False,
                    ) as data:
                        metadata = json.loads(
                            str(data["metadata"].item())
                        )

                    target_kics.add(str(metadata["kic"]))

                except (
                    OSError,
                    KeyError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    pass

        map_paths.extend(target_paths)
        sample_ids.update(target_sample_ids)
        kics.update(target_kics)

        target_records.append({
            "analysis_id": analysis_id,
            "target": target,
            "method": method,
            "map_count": len(target_paths),
            "unique_sample_count": len(target_sample_ids),
            "unique_kic_count": len(target_kics),
            "max_samples": row.get("max_samples"),
            "output_dir": str(output_dir),
            "analysis_output_dir": str(target_output_dir),
            "copied_map_count": len(target_paths),
        })

    record = {
        "analysis_id": analysis_id,
        "model": row.get("model"),
        "input": row.get("mode"),
        "duration": row.get("duration"),
        "targets": "+".join(targets),
        "method": method,
        "status": status.get("status", "not_started"),
        "return_code": status.get("return_code"),
        "complete": (
            status.get("status") == "completed"
            and status.get("return_code") == 0
            and bool(map_paths)
        ),
        "map_count": len(map_paths),
        "unique_sample_count": len(sample_ids),
        "unique_kic_count": len(kics),
        "max_samples": row.get("max_samples"),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "analysis_output_dir": str(
            interpretation_dir / analysis_id
        ),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "reason": status.get("reason"),
    }

    return record, target_records


# ============================================================
# Load attribution
# ============================================================

def load_attribution(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        metadata = json.loads(
            str(data["metadata"].item())
        )

        return {
            "attribution": np.asarray(
                data["attribution"],
                dtype=np.float64,
            ).reshape(-1),

            "input_axis": np.asarray(
                data["input_axis"],
                dtype=np.float64,
            ).reshape(-1),

            "signal": np.asarray(
                data["signal"],
                dtype=np.float64,
            ).reshape(-1),

            "model_input": np.asarray(
                data["model_input"],
                dtype=np.float64,
            ),

            "metadata": metadata,
        }


# ============================================================
# Agreement metrics
# ============================================================

def top_fraction_overlap(first, second, fraction=0.1):
    first = np.asarray(first).reshape(-1)
    second = np.asarray(second).reshape(-1)

    if len(first) != len(second):
        raise ValueError("Attribution lengths differ.")

    if len(first) == 0:
        return np.nan

    count = max(
        1,
        math.ceil(len(first) * fraction),
    )

    first_top = set(
        np.argpartition(
            np.abs(first),
            -count,
        )[-count:]
    )

    second_top = set(
        np.argpartition(
            np.abs(second),
            -count,
        )[-count:]
    )

    return float(
        len(first_top & second_top) / count
    )


def cosine_similarity(first, second, absolute=False):
    first = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)

    second = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if len(first) != len(second):
        raise ValueError("Attribution lengths differ.")

    if absolute:
        first = np.abs(first)
        second = np.abs(second)

    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    return float(
        np.dot(first, second) / denominator
    )


# ============================================================
# PSD frequency localisation
# ============================================================

def frequency_localisation(attribution, frequency, nu_max):
    """
    Frequency localisation of discrete per-feature attribution.

    Absolute, positive and negative attribution are analysed
    separately.

    Centroid and width use log10(frequency).

    Fractions are sums over input features; they are not weighted
    by linear-frequency bin widths.

    Exploratory intervals:
        wide:   0.50 - 1.50 * nu_max
        medium: 0.75 - 1.25 * nu_max
        narrow: 0.90 - 1.10 * nu_max
    """

    attr = np.asarray(
        attribution,
        dtype=np.float64,
    ).reshape(-1)

    freq = np.asarray(
        frequency,
        dtype=np.float64,
    ).reshape(-1)

    if len(attr) != len(freq):
        raise ValueError(
            "Attribution and frequency lengths differ."
        )

    valid = (
        np.isfinite(attr)
        & np.isfinite(freq)
        & (freq > 0)
    )

    attr = attr[valid]
    freq = freq[valid]

    bands = {
        "wide": (0.50, 1.50),
        "medium": (0.75, 1.25),
        "narrow": (0.90, 1.10),
    }

    result = {
        "total_absolute_attribution": np.nan,
        "total_positive_attribution": np.nan,
        "total_negative_attribution_magnitude": np.nan,
        "attribution_centroid_log10_frequency": np.nan,
        "attribution_centroid_frequency": np.nan,
        "attribution_width_dex": np.nan,
        "peak_attribution_frequency": np.nan,
        "peak_attribution_nu_max_ratio": np.nan,
        "attribution_centroid_nu_max_ratio": np.nan,
        "attribution_entropy": np.nan,
        "attribution_entropy_normalised": np.nan,
        "attribution_effective_bins": np.nan,
    }

    for name in bands:
        result[f"envelope_fraction_{name}"] = np.nan
        result[f"outside_envelope_fraction_{name}"] = np.nan
        result[f"positive_envelope_fraction_{name}"] = np.nan
        result[f"negative_envelope_fraction_{name}"] = np.nan

    if len(attr) == 0:
        return result

    absolute = np.abs(attr)
    positive = np.clip(attr, 0, None)
    negative = np.clip(-attr, 0, None)

    total_abs = float(absolute.sum())
    total_pos = float(positive.sum())
    total_neg = float(negative.sum())

    result["total_absolute_attribution"] = total_abs
    result["total_positive_attribution"] = total_pos
    result["total_negative_attribution_magnitude"] = total_neg

    if total_abs <= 0 or not np.isfinite(total_abs):
        return result

    probabilities = absolute / total_abs
    log_freq = np.log10(freq)

    centroid_log = float(
        np.sum(probabilities * log_freq)
    )

    width_dex = float(
        np.sqrt(
            np.sum(
                probabilities
                * (log_freq - centroid_log) ** 2
            )
        )
    )

    centroid_frequency = float(10 ** centroid_log)

    peak_frequency = float(
        freq[np.argmax(absolute)]
    )

    nonzero = probabilities > 0

    entropy = float(
        -np.sum(
            probabilities[nonzero]
            * np.log(probabilities[nonzero])
        )
    )

    entropy_normalised = (
        float(entropy / np.log(len(probabilities)))
        if len(probabilities) > 1
        else 0.0
    )

    result.update({
        "attribution_centroid_log10_frequency": centroid_log,
        "attribution_centroid_frequency": centroid_frequency,
        "attribution_width_dex": width_dex,
        "peak_attribution_frequency": peak_frequency,
        "attribution_entropy": entropy,
        "attribution_entropy_normalised": entropy_normalised,
        "attribution_effective_bins": float(np.exp(entropy)),
    })

    nu_max = safe_float(nu_max)

    if not np.isfinite(nu_max) or nu_max <= 0:
        return result

    result["peak_attribution_nu_max_ratio"] = (
        peak_frequency / nu_max
    )

    result["attribution_centroid_nu_max_ratio"] = (
        centroid_frequency / nu_max
    )

    ratio = freq / nu_max

    for name, (lower, upper) in bands.items():
        mask = (
            (ratio >= lower)
            & (ratio <= upper)
        )

        fraction = float(
            absolute[mask].sum() / total_abs
        )

        result[f"envelope_fraction_{name}"] = fraction

        result[f"outside_envelope_fraction_{name}"] = (
            1.0 - fraction
        )

        if total_pos > 0:
            result[
                f"positive_envelope_fraction_{name}"
            ] = float(
                positive[mask].sum() / total_pos
            )

        if total_neg > 0:
            result[
                f"negative_envelope_fraction_{name}"
            ] = float(
                negative[mask].sum() / total_neg
            )

    return result


# ============================================================
# Plot paired attributions
# ============================================================

def smooth_for_plot(values):
    values = np.asarray(values)

    if len(values) < 3:
        return values

    window = min(
        9,
        max(3, len(values) // 500),
    )

    window = min(
        window,
        len(values)
        if len(values) % 2
        else len(values) - 1,
    )

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="same",
    )


def plot_hump_localisation(
    pair,
    output_dir,
    analysis_id,
    sample_id,
    target,
    nu_max,
):
    if not np.isfinite(nu_max) or nu_max <= 0:
        return

    frequency = pair["ig"]["input_axis"]
    ig = pair["ig"]["attribution"]
    smoothgrad = pair["smoothgrad"]["attribution"]

    frequency_ratio = frequency / nu_max
    valid = (
        np.isfinite(frequency_ratio)
        & (frequency_ratio > 0)
        & np.isfinite(ig)
        & np.isfinite(smoothgrad)
    )

    if not valid.any():
        return

    frequency_ratio = frequency_ratio[valid]
    ig_absolute = np.abs(ig[valid])
    smoothgrad_absolute = np.abs(smoothgrad[valid])

    ig_scale = max(float(ig_absolute.max()), np.finfo(float).tiny)
    smoothgrad_scale = max(
        float(smoothgrad_absolute.max()),
        np.finfo(float).tiny,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for lower, upper, label, alpha in (
        (0.50, 1.50, "Wide (0.50-1.50 x nu_max)", 0.08),
        (0.75, 1.25, "Medium (0.75-1.25 x nu_max)", 0.12),
        (0.90, 1.10, "Narrow (0.90-1.10 x nu_max)", 0.18),
    ):
        ax.axvspan(
            lower,
            upper,
            color="#2ca02c",
            alpha=alpha,
            label=label,
        )

    ax.plot(
        frequency_ratio,
        smooth_for_plot(ig_absolute / ig_scale),
        color="#d62728",
        label="Integrated Gradients",
        linewidth=1,
    )

    ax.plot(
        frequency_ratio,
        smooth_for_plot(smoothgrad_absolute / smoothgrad_scale),
        color="#1f77b4",
        label="SmoothGrad",
        linewidth=1,
    )

    ax.axvline(
        1,
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"Catalog $\nu_{\max}$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Frequency / catalog $\nu_{\max}$")
    ax.set_ylabel("Absolute attribution (normalised to each method's maximum)")
    ax.set_title(
        f"{analysis_id} | {sample_id} | {target}: "
        "attribution localisation around the oscillation hump"
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"kic_{sample_id}_hump_localisation.png",
        dpi=160,
    )
    plt.close(fig)


def plot_attribution_pair(
    pair,
    output_dir,
    analysis_id,
    sample_id,
    target,
    mode,
):
    axis = pair["ig"]["input_axis"]
    signal = pair["ig"]["signal"]

    ig = pair["ig"]["attribution"]
    sg = pair["smoothgrad"]["attribution"]

    ig_scale = max(
        float(np.max(np.abs(ig))),
        np.finfo(float).tiny,
    )

    sg_scale = max(
        float(np.max(np.abs(sg))),
        np.finfo(float).tiny,
    )

    ig_scaled = ig / ig_scale
    sg_scaled = sg / sg_scale

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Overlay
    fig, signal_ax = plt.subplots(
        figsize=(12, 4.5)
    )

    signal_ax.plot(
        axis,
        signal,
        color="black",
        alpha=0.55,
        linewidth=0.8,
    )

    signal_ax.set_xlabel(
        "Frequency (µHz)"
        if mode == "psd"
        else "Time (days)"
    )

    signal_ax.set_ylabel("Input signal")

    if mode == "psd":
        signal_ax.set_xscale("log")

        nu_max = safe_float(
            pair["ig"]["metadata"].get("nu_max")
        )

        if np.isfinite(nu_max) and nu_max > 0:
            signal_ax.axvline(
                nu_max,
                color="green",
                linestyle="--",
                label=r"True $\nu_{\max}$",
            )

    attribution_ax = signal_ax.twinx()

    attribution_ax.plot(
        axis,
        smooth_for_plot(ig_scaled),
        color="#d62728",
        label="Integrated Gradients",
    )

    attribution_ax.plot(
        axis,
        smooth_for_plot(sg_scaled),
        color="#1f77b4",
        alpha=0.85,
        label="SmoothGrad",
    )

    attribution_ax.axhline(
        0,
        color="gray",
        linewidth=0.6,
    )

    attribution_ax.set_ylabel(
        "Independently normalised attribution"
    )

    handles1, labels1 = (
        signal_ax.get_legend_handles_labels()
    )

    handles2, labels2 = (
        attribution_ax.get_legend_handles_labels()
    )

    signal_ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    signal_ax.set_title(
        f"{analysis_id} | {sample_id} | "
        f"{target}: attribution overlay"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / f"kic_{sample_id}_overlay.png",
        dpi=160,
    )

    plt.close(fig)

    # Signed scatter
    sample_count = min(len(ig_scaled), 10000)

    if sample_count < len(ig_scaled):
        indices = np.sort(
            np.random.default_rng(42).choice(
                len(ig_scaled),
                sample_count,
                replace=False,
            )
        )

    else:
        indices = np.arange(len(ig_scaled))

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.scatter(
        ig_scaled[indices],
        sg_scaled[indices],
        s=7,
        alpha=0.25,
        edgecolors="none",
    )

    ax.plot(
        [-1, 1],
        [-1, 1],
        "--",
        color="gray",
        linewidth=1,
    )

    ax.set(
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
    )

    ax.set_xlabel("Integrated Gradients (normalised)")
    ax.set_ylabel("SmoothGrad (normalised)")

    ax.set_title(
        f"{analysis_id} | {sample_id} | "
        f"{target}: signed attribution"
    )

    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        output_dir / f"kic_{sample_id}_scatter.png",
        dpi=160,
    )

    plt.close(fig)

    if mode == "psd":
        plot_hump_localisation(
            pair,
            output_dir,
            analysis_id,
            sample_id,
            target,
            safe_float(pair["ig"]["metadata"].get("nu_max")),
        )


# ============================================================
# Compare one attribution pair
# ============================================================

def compare_attributions(
    ig_path,
    smoothgrad_path,
    output_dir,
    analysis_id,
    model,
    mode,
    duration,
):
    ig = load_attribution(ig_path)
    sg = load_attribution(smoothgrad_path)

    ig_metadata = ig["metadata"]
    sg_metadata = sg["metadata"]

    if ig_metadata.get("method") != "ig":
        raise ValueError(f"Invalid IG metadata: {ig_path}")

    if sg_metadata.get("method") != "smoothgrad":
        raise ValueError(
            f"Invalid SmoothGrad metadata: {smoothgrad_path}"
        )

    for key in (
        "kic",
        "target",
        "checkpoint_path",
        "input_mode",
        "model_type",
    ):
        if ig_metadata.get(key) != sg_metadata.get(key):
            raise ValueError(
                f"Metadata mismatch for {key}: "
                f"{ig_path}, {smoothgrad_path}"
            )

    if ig_metadata.get("input_mode") != mode:
        raise ValueError(
            f"Plan/input-mode mismatch for {ig_path}"
        )

    for key in ("sample_id", "duration"):
        first_value = ig_metadata.get(key)
        second_value = sg_metadata.get(key)

        if first_value is not None and second_value is not None:
            if first_value != second_value:
                raise ValueError(
                    f"Metadata mismatch for {key}: "
                    f"{ig_path}, {smoothgrad_path}"
                )

    ig_nu_max = safe_float(
        ig_metadata.get("nu_max")
    )

    sg_nu_max = safe_float(
        sg_metadata.get("nu_max")
    )

    if (
        np.isfinite(ig_nu_max)
        and np.isfinite(sg_nu_max)
        and not np.isclose(
            ig_nu_max,
            sg_nu_max,
            rtol=1e-8,
            atol=0,
        )
    ):
        raise ValueError(
            f"nu_max mismatch: {ig_path}, {smoothgrad_path}"
        )

    nu_max = (
        ig_nu_max
        if np.isfinite(ig_nu_max)
        else sg_nu_max
    )

    first = ig["attribution"]
    second = sg["attribution"]

    if len(first) == 0:
        raise ValueError("Empty attribution array.")

    if not (
        len(first)
        == len(second)
        == len(ig["input_axis"])
        == len(ig["signal"])
        == len(sg["input_axis"])
        == len(sg["signal"])
    ):
        raise ValueError(
            f"Attribution lengths do not align: "
            f"{ig_path}, {smoothgrad_path}"
        )

    if not np.array_equal(
        ig["input_axis"],
        sg["input_axis"],
    ):
        raise ValueError("Frequency/time axes differ.")

    if not np.array_equal(
        ig["model_input"],
        sg["model_input"],
    ):
        raise ValueError("Model inputs differ.")

    if not np.array_equal(
        ig["signal"],
        sg["signal"],
    ):
        raise ValueError("Plotting signals differ.")

    if not (
        np.isfinite(first).all()
        and np.isfinite(second).all()
    ):
        raise ValueError("Non-finite attribution values.")

    pearson = pd.Series(first).corr(
        pd.Series(second),
        method="pearson",
    )

    spearman_abs = pd.Series(
        np.abs(first)
    ).corr(
        pd.Series(np.abs(second)),
        method="spearman",
    )

    kic = str(ig_metadata["kic"])
    target = str(ig_metadata["target"])

    sample_id = str(
        ig_metadata.get("sample_id") or kic
    )

    plot_attribution_pair(
        {"ig": ig, "smoothgrad": sg},
        output_dir / target,
        analysis_id,
        sample_id,
        target,
        mode,
    )

    localisation = {}

    if mode == "psd":
        ig_loc = frequency_localisation(
            first,
            ig["input_axis"],
            nu_max,
        )

        sg_loc = frequency_localisation(
            second,
            sg["input_axis"],
            nu_max,
        )

        localisation.update({
            f"ig_{key}": value
            for key, value in ig_loc.items()
        })

        localisation.update({
            f"smoothgrad_{key}": value
            for key, value in sg_loc.items()
        })

        if not np.isfinite(nu_max) or nu_max <= 0:
            print(
                f"Warning: {sample_id}, {target}: "
                "nu_max unavailable; relative localisation is NaN."
            )

    return {
        "analysis_id": analysis_id,
        "model": model,
        "input": mode,
        "duration": duration,
        "target": target,
        "kic": kic,
        "sample_id": sample_id,
        "nu_max": nu_max,
        "n_bins": len(first),

        "pearson_signed": pearson,
        "spearman_absolute": spearman_abs,

        "cosine_signed": cosine_similarity(
            first,
            second,
            absolute=False,
        ),

        "cosine_absolute": cosine_similarity(
            first,
            second,
            absolute=True,
        ),

        "top_10pct_overlap": top_fraction_overlap(
            first,
            second,
        ),

        "ig_n_steps": ig_metadata.get("n_steps"),

        "smoothgrad_n_samples": sg_metadata.get(
            "n_samples"
        ),

        "smoothgrad_noise_std": sg_metadata.get(
            "noise_std"
        ),

        "checkpoint_path": ig_metadata.get(
            "checkpoint_path"
        ),

        "ig_file": str(ig_path),
        "smoothgrad_file": str(smoothgrad_path),

        **localisation,
    }


# ============================================================
# Collect IG / SmoothGrad pairs
# ============================================================

def collect_agreement(jobs, status_dir, agreement_dir):
    groups = {}

    for row in jobs:
        method = (
            row.get("method") or ""
        ).strip().lower()

        if method not in {"ig", "smoothgrad"}:
            continue

        analysis_id = row["analysis_id"]

        suffixes = (
            ("_ig",)
            if method == "ig"
            else ("_smoothgrad", "_sg")
        )

        pair_id = next(
            (
                analysis_id[:-len(suffix)]
                for suffix in suffixes
                if analysis_id.endswith(suffix)
            ),
            analysis_id,
        )

        run_dir = resolve_path(row["run_dir"])

        status = read_json(
            status_dir / f"{analysis_id}.json"
        )

        output_dir = status_output_dir(
            status,
            run_dir,
        )

        group = groups.setdefault(pair_id, {})

        if method in group:
            raise ValueError(
                f"Duplicate {method} job for {pair_id}"
            )

        group[method] = {
            "row": row,
            "run_dir": run_dir,
            "output_dir": output_dir,
        }

    records = []

    for pair_id, methods in groups.items():
        if not {"ig", "smoothgrad"}.issubset(methods):
            continue

        ig_job = methods["ig"]
        sg_job = methods["smoothgrad"]

        ig_row = ig_job["row"]
        sg_row = sg_job["row"]

        for field in (
            "model",
            "mode",
            "duration",
            "targets",
        ):
            if ig_row.get(field) != sg_row.get(field):
                raise ValueError(
                    f"{pair_id}: mismatch in {field}"
                )

        if ig_job["run_dir"] != sg_job["run_dir"]:
            raise ValueError(
                f"{pair_id}: different training run directories."
            )

        ig_dir = ig_job["output_dir"]
        sg_dir = sg_job["output_dir"]

        ig_paths = {
            path.stem[:-len("_ig")]: path
            for path in ig_dir.glob("kic_*_ig.npz")
            if path.is_file()
        }

        sg_paths = {
            path.stem[:-len("_smoothgrad")]: path
            for path in sg_dir.glob("kic_*_smoothgrad.npz")
            if path.is_file()
        }

        common_stems = sorted(
            ig_paths.keys() & sg_paths.keys()
        )

        if not common_stems:
            print(
                f"Warning: no matching files for {pair_id}"
            )
            continue

        pair_output_dir = agreement_dir / pair_id

        for stem in common_stems:
            records.append(
                compare_attributions(
                    ig_path=ig_paths[stem],
                    smoothgrad_path=sg_paths[stem],
                    output_dir=pair_output_dir,
                    analysis_id=pair_id,
                    model=ig_row.get("model"),
                    mode=ig_row.get("mode"),
                    duration=ig_row.get("duration"),
                )
            )

        print(
            f"Paired attribution arrays for "
            f"{pair_id}: {len(common_stems)}"
        )

    return records


# ============================================================
# Summary heatmaps
# ============================================================

def plot_summary_heatmap(
    summary,
    metric,
    title,
    limits,
    cmap,
    plots_dir,
):
    if metric not in summary.columns:
        return

    matrix = summary.pivot_table(
        index="analysis_id",
        columns="target",
        values=metric,
        aggfunc="first",
        dropna=False,
    )

    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).any():
        return

    fig, ax = plt.subplots(
        figsize=(
            max(7, matrix.shape[1] * 2.0),
            max(4, matrix.shape[0] * 0.65),
        )
    )

    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        aspect="auto",
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[
                row_index,
                column_index,
            ]

            if pd.notna(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        matrix.columns,
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Prediction target")
    ax.set_ylabel("Interpretation run")
    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label=metric.replace("_", " "),
    )

    fig.tight_layout()

    plot_path = (
        plots_dir
        / f"interpretation_agreement_{metric}.png"
    )

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved {plot_path}")


# ============================================================
# Agreement and localisation summaries
# ============================================================

def write_agreement_summary(
    records,
    output_dir,
    plots_dir,
):
    per_star_path = (
        output_dir
        / "interpretation_agreement_per_star.csv"
    )

    summary_path = (
        output_dir
        / "interpretation_agreement_summary.csv"
    )

    per_star = pd.DataFrame(records)
    per_star.to_csv(per_star_path, index=False)

    if per_star.empty:
        pd.DataFrame(columns=[
            "analysis_id",
            "model",
            "input",
            "duration",
            "target",
            "n_stars",
        ]).to_csv(summary_path, index=False)

        print("No paired IG/SmoothGrad files found.")
        return

    group_columns = [
        "analysis_id",
        "model",
        "input",
        "duration",
        "target",
    ]

    agreement_metrics = [
        "pearson_signed",
        "spearman_absolute",
        "cosine_signed",
        "cosine_absolute",
        "top_10pct_overlap",
    ]

    localisation_metrics = []

    for method in ("ig", "smoothgrad"):
        localisation_metrics.extend([
            f"{method}_attribution_centroid_frequency",
            f"{method}_attribution_centroid_nu_max_ratio",
            f"{method}_attribution_width_dex",
            f"{method}_peak_attribution_frequency",
            f"{method}_peak_attribution_nu_max_ratio",
            f"{method}_attribution_entropy_normalised",
            f"{method}_attribution_effective_bins",
        ])

        for band in ("wide", "medium", "narrow"):
            localisation_metrics.extend([
                f"{method}_envelope_fraction_{band}",
                f"{method}_outside_envelope_fraction_{band}",
                f"{method}_positive_envelope_fraction_{band}",
                f"{method}_negative_envelope_fraction_{band}",
            ])

    metric_columns = [
        metric
        for metric in agreement_metrics + localisation_metrics
        if metric in per_star.columns
    ]

    for metric in metric_columns:
        per_star[metric] = pd.to_numeric(
            per_star[metric],
            errors="coerce",
        )

    aggregation = {
        "n_stars": ("kic", "nunique"),
        "n_samples": ("sample_id", "count"),
    }

    for metric in metric_columns:
        aggregation[f"median_{metric}"] = (
            metric,
            "median",
        )

        aggregation[f"mean_{metric}"] = (
            metric,
            "mean",
        )

        aggregation[f"std_{metric}"] = (
            metric,
            "std",
        )

        aggregation[f"n_valid_{metric}"] = (
            metric,
            "count",
        )

    summary = (
        per_star
        .groupby(group_columns, dropna=False)
        .agg(**aggregation)
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    heatmaps = [
        (
            "median_pearson_signed",
            "Median signed Pearson correlation",
            (-1, 1),
            "coolwarm",
        ),
        (
            "median_spearman_absolute",
            "Median absolute-attribution Spearman correlation",
            (-1, 1),
            "coolwarm",
        ),
        (
            "median_cosine_signed",
            "Median signed cosine similarity",
            (-1, 1),
            "coolwarm",
        ),
        (
            "median_cosine_absolute",
            "Median absolute cosine similarity",
            (0, 1),
            "YlGnBu",
        ),
        (
            "median_top_10pct_overlap",
            "Median top-10% attribution overlap",
            (0, 1),
            "YlGnBu",
        ),
    ]

    for method in ("ig", "smoothgrad"):
        for band in ("wide", "medium", "narrow"):
            heatmaps.append((
                f"median_{method}_envelope_fraction_{band}",
                f"{method.upper()} attribution fraction: "
                f"{band} envelope",
                (0, 1),
                "YlGnBu",
            ))

        heatmaps.extend([
            (
                f"median_{method}_attribution_entropy_normalised",
                f"{method.upper()} attribution entropy",
                (0, 1),
                "viridis",
            ),
            (
                f"median_{method}_attribution_centroid_nu_max_ratio",
                rf"{method.upper()} centroid / $\nu_{{\max}}$",
                (0, 3),
                "viridis",
            ),
        ])

    for metric, title, limits, cmap in heatmaps:
        plot_summary_heatmap(
            summary,
            metric,
            title,
            limits,
            cmap,
            plots_dir,
        )

    # Distributions across stars
    boxplot_metrics = [
        "cosine_signed",
        "cosine_absolute",
        "top_10pct_overlap",
        "ig_envelope_fraction_wide",
        "smoothgrad_envelope_fraction_wide",
        "ig_envelope_fraction_narrow",
        "smoothgrad_envelope_fraction_narrow",
        "ig_attribution_entropy_normalised",
        "smoothgrad_attribution_entropy_normalised",
        "ig_attribution_width_dex",
        "smoothgrad_attribution_width_dex",
    ]

    for metric in boxplot_metrics:
        if metric not in per_star.columns:
            continue

        for target, target_df in per_star.groupby("target"):
            labels = []
            values = []

            for analysis_id, group in target_df.groupby(
                "analysis_id"
            ):
                valid_values = (
                    group[metric]
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(valid_values) == 0:
                    continue

                labels.append(str(analysis_id))
                values.append(valid_values)

            if not values:
                continue

            fig, ax = plt.subplots(
                figsize=(
                    max(8, len(labels) * 1.3),
                    5.5,
                )
            )

            ax.boxplot(
                values,
                tick_labels=labels,
                showfliers=False,
            )

            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xlabel("Interpretation run")
            ax.set_title(
                f"{target}: distribution of {metric}"
            )

            ax.tick_params(
                axis="x",
                labelrotation=45,
            )

            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()

            safe_target = str(target).replace("/", "_")

            plot_path = (
                plots_dir
                / f"interpretation_boxplot_{safe_target}_{metric}.png"
            )

            fig.savefig(plot_path, dpi=180)
            plt.close(fig)

    print(f"Saved {per_star_path}")
    print(f"Saved {summary_path}")


# ============================================================
# Coverage plot
# ============================================================

def plot_map_coverage(
    runs_df,
    target_records,
    plots_dir,
):
    if runs_df.empty or not target_records:
        return

    coverage_df = pd.DataFrame(target_records)

    coverage_df["max_samples"] = pd.to_numeric(
        coverage_df["max_samples"],
        errors="coerce",
    )

    coverage_df["coverage"] = (
        coverage_df["map_count"]
        / coverage_df["max_samples"].replace(0, np.nan)
    )

    analysis_order = runs_df["analysis_id"].tolist()

    target_order = (
        coverage_df["target"]
        .drop_duplicates()
        .tolist()
    )

    coverage = coverage_df.pivot(
        index="analysis_id",
        columns="target",
        values="coverage",
    ).reindex(
        index=analysis_order,
        columns=target_order,
    )

    counts = coverage_df.pivot(
        index="analysis_id",
        columns="target",
        values="map_count",
    ).reindex(
        index=analysis_order,
        columns=target_order,
    )

    caps = coverage_df.pivot(
        index="analysis_id",
        columns="target",
        values="max_samples",
    ).reindex(
        index=analysis_order,
        columns=target_order,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(target_order) * 2.2),
            max(4.5, len(analysis_order) * 0.65),
        )
    )

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e5e7eb")

    image = ax.imshow(
        coverage.to_numpy(dtype=float),
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    for row_index, analysis_id in enumerate(analysis_order):
        for column_index, target in enumerate(target_order):
            count = counts.loc[analysis_id, target]
            cap = caps.loc[analysis_id, target]

            if pd.isna(count):
                continue

            if pd.notna(cap) and cap > 0:
                label = f"{int(count)} / {int(cap)}"

                value = coverage.loc[
                    analysis_id,
                    target,
                ]

                text_color = (
                    "white"
                    if pd.notna(value) and value >= 0.55
                    else "#17212b"
                )

            else:
                label = f"{int(count)}"
                text_color = "#17212b"

            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    status_by_analysis = (
        runs_df
        .set_index("analysis_id")["status"]
    )

    ax.set_yticks(range(len(analysis_order)))

    ax.set_yticklabels([
        f"{analysis_id} ({status_by_analysis[analysis_id]})"
        for analysis_id in analysis_order
    ])

    ax.set_xticks(range(len(target_order)))
    ax.set_xticklabels(target_order)

    ax.set_xlabel("Prediction target")
    ax.set_ylabel("Analysis (run status)")
    ax.set_title("Saliency-map coverage by target")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        "Fraction of requested map cap"
    )

    colorbar.set_ticks([
        0, 0.25, 0.5, 0.75, 1
    ])

    colorbar.set_ticklabels([
        "0%", "25%", "50%", "75%", "100%"
    ])

    ax.set_xticks(
        [
            position - 0.5
            for position in range(1, len(target_order))
        ],
        minor=True,
    )

    ax.set_yticks(
        [
            position - 0.5
            for position in range(1, len(analysis_order))
        ],
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    fig.tight_layout()

    plot_path = (
        plots_dir / "interpretation_map_coverage.png"
    )

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved {plot_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
    )

    parser.add_argument(
        "--status-dir",
        type=Path,
        default=DEFAULT_STATUS_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    status_dir = args.status_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    plots_dir = output_dir / "plots"
    interpretation_dir = output_dir / "interpretation"
    agreement_dir = output_dir / AGREEMENT_DIR_NAME

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with plan_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as plan_file:
        jobs = list(csv.DictReader(plan_file))

    records = []
    target_records = []

    for row in jobs:
        record, per_target = collect_run_record(
            row,
            status_dir,
            interpretation_dir,
        )

        records.append(record)
        target_records.extend(per_target)

    runs_df = pd.DataFrame(records)

    runs_path = (
        output_dir / "interpretation_runs.csv"
    )

    targets_path = (
        output_dir / "interpretation_target_coverage.csv"
    )

    runs_df.to_csv(
        runs_path,
        index=False,
    )

    pd.DataFrame(
        target_records
    ).to_csv(
        targets_path,
        index=False,
    )

    agreement_records = collect_agreement(
        jobs,
        status_dir,
        agreement_dir,
    )

    write_agreement_summary(
        agreement_records,
        output_dir,
        plots_dir,
    )

    plot_map_coverage(
        runs_df,
        target_records,
        plots_dir,
    )

    print(f"Saved {runs_path}")
    print(f"Saved {targets_path}")

    if not runs_df.empty:
        completed = int(
            runs_df["complete"].sum()
        )

        print(
            f"Completed analyses with maps: "
            f"{completed}/{len(runs_df)}"
        )

    print("\nInterpretation analysis complete.")


if __name__ == "__main__":
    main()
