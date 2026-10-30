
#!/usr/bin/env python3
"""
Analyse the outputs of cross_validate.py.

Usage:
    python -m Baseline.analysis.analyze_cross_validation
    python -m Baseline.analysis.analyze_cross_validation \
        --run-dir Baseline/experiments/01_architecture/A1_resnet1d_lc_80d/cv_runs/20261030_...

Optional:
    --manifest Baseline/manifest.csv
    --output-root Baseline/results_analysis/cross_validation

Outputs:
    regression/
        <target>_predicted_vs_true.png
        <target>_residuals.png
        <target>_residuals_vs_true.png
        <target>_physical_predicted_vs_true.png  (nu_max, delta_nu)
    classification/
        confusion_matrix.png
        roc_curve.png
        class_probabilities.png
    training/
        loss_curves.png
        best_epochs.png
    fold_comparison/
        fold_metrics.png
    pooled_metrics.json
    analysis_summary.json
    README.md

This script does not train or load models. It analyses the saved
out-of-fold predictions and training histories. By default, it discovers
and analyzes every completed run under Baseline/experiments.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Baseline.paths import BASELINE_DIR
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
    roc_auc_score,
    classification_report,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REGRESSION_TARGETS = {"nu_max", "delta_nu", "metallicity"}

PHYSICAL_TARGETS = {
    "nu_max": {
        "label": r"$\nu_{\max}$",
        "unit": r"$\mu$Hz",
    },
    "delta_nu": {
        "label": r"$\Delta\nu$",
        "unit": r"$\mu$Hz",
    },
}

STELLAR_PARAMETERS = [
    "teff",
    "metallicity",
    "fe_h",
    "logg",
    "mass",
    "radius",
    "evolstate",
]

DEFAULT_RUNS_ROOT = BASELINE_DIR / "experiments"
DEFAULT_OUTPUT_ROOT = (
    BASELINE_DIR / "results_analysis" / "cross_validation"
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def save_json(data, path):
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def safe_float(value):
    """Convert values to JSON-safe floats."""
    value = float(value)
    return value if np.isfinite(value) else None


def get_target_columns(frame, target):
    truth_col = f"truth_{target}"
    pred_col = f"pred_{target}"

    if truth_col not in frame.columns:
        raise ValueError(f"Missing column: {truth_col}")

    if pred_col not in frame.columns:
        raise ValueError(f"Missing column: {pred_col}")

    truth = pd.to_numeric(frame[truth_col], errors="coerce").to_numpy()
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy()

    valid = np.isfinite(truth) & np.isfinite(pred)

    return truth[valid], pred[valid], valid


def regression_metrics(truth, predictions):
    truth = np.asarray(truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    valid = np.isfinite(truth) & np.isfinite(predictions)
    truth = truth[valid]
    predictions = predictions[valid]

    if len(truth) == 0:
        return {
            "n": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "bias": None,
        }

    residuals = predictions - truth
    mse = mean_squared_error(truth, predictions)

    return {
        "n": int(len(truth)),
        "mae": safe_float(mean_absolute_error(truth, predictions)),
        "mse": safe_float(mse),
        "rmse": safe_float(np.sqrt(mse)),
        "bias": safe_float(np.mean(residuals)),
    }


def physical_values(target, values):
    """
    Convert log10 target values to physical units where appropriate.

    This assumes the dataset returns log10(nu_max) and
    log10(delta_nu), as in the current pipeline.
    """
    values = np.asarray(values, dtype=float)

    if target in ("nu_max", "delta_nu"):
        return np.power(10.0, values)

    return values


def target_axis_label(target, physical=False):
    if physical and target in PHYSICAL_TARGETS:
        info = PHYSICAL_TARGETS[target]
        return f"{info['label']} ({info['unit']})"

    if target == "nu_max":
        return r"$\log_{10}(\nu_{\max})$"

    if target == "delta_nu":
        return r"$\log_{10}(\Delta\nu)$"

    return target


# ---------------------------------------------------------------------
# Regression plots
# ---------------------------------------------------------------------

def plot_predicted_vs_true(
    truth,
    predictions,
    target,
    output_path,
    physical=False,
):
    if physical:
        truth = physical_values(target, truth)
        predictions = physical_values(target, predictions)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        truth,
        predictions,
        alpha=0.45,
        s=14,
        edgecolors="none",
    )

    lower = min(np.min(truth), np.min(predictions))
    upper = max(np.max(truth), np.max(predictions))

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        label="Ideal prediction",
    )

    ax.set_xlabel(f"True {target_axis_label(target, physical)}")
    ax.set_ylabel(f"Predicted {target_axis_label(target, physical)}")
    ax.set_title(f"{target}: predicted vs. true")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_distribution(
    truth,
    predictions,
    target,
    output_path,
    physical=False,
):
    if physical:
        truth = physical_values(target, truth)
        predictions = physical_values(target, predictions)

    residuals = predictions - truth

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        residuals,
        bins=40,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.axvline(
        np.mean(residuals),
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {np.mean(residuals):.4g}",
    )

    ax.set_xlabel(
        f"Prediction residual ({target_axis_label(target, physical)})"
    )
    ax.set_ylabel("Number of stars")
    ax.set_title(f"{target}: residual distribution")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residuals_vs_true(
    truth,
    predictions,
    target,
    output_path,
    physical=False,
):
    if physical:
        truth = physical_values(target, truth)
        predictions = physical_values(target, predictions)

    residuals = predictions - truth

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        truth,
        residuals,
        alpha=0.45,
        s=14,
        edgecolors="none",
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel(f"True {target_axis_label(target, physical)}")
    ax.set_ylabel("Prediction − true")
    ax.set_title(f"{target}: residuals vs. true value")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyse_regression(predictions, targets, output_dir):
    regression_dir = output_dir / "regression"
    regression_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for target in targets:
        if target not in REGRESSION_TARGETS:
            continue

        truth, pred, _ = get_target_columns(predictions, target)

        target_dir = regression_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        results[target] = {
            "target_space": regression_metrics(truth, pred),
        }

        plot_predicted_vs_true(
            truth,
            pred,
            target,
            target_dir / "predicted_vs_true.png",
        )

        plot_residual_distribution(
            truth,
            pred,
            target,
            target_dir / "residual_distribution.png",
        )

        plot_residuals_vs_true(
            truth,
            pred,
            target,
            target_dir / "residuals_vs_true.png",
        )

        if target in PHYSICAL_TARGETS:
            physical_truth = physical_values(target, truth)
            physical_pred = physical_values(target, pred)

            results[target]["physical_space"] = regression_metrics(
                physical_truth,
                physical_pred,
            )

            plot_predicted_vs_true(
                truth,
                pred,
                target,
                target_dir / "physical_predicted_vs_true.png",
                physical=True,
            )

            plot_residual_distribution(
                truth,
                pred,
                target,
                target_dir / "physical_residual_distribution.png",
                physical=True,
            )

            plot_residuals_vs_true(
                truth,
                pred,
                target,
                target_dir / "physical_residuals_vs_true.png",
                physical=True,
            )

    return results


# ---------------------------------------------------------------------
# Classification plots and metrics
# ---------------------------------------------------------------------

def analyse_classification(predictions, output_dir):
    required = {
        "truth_class",
        "pred_class",
        "pred_class_probability",
        "pred_class_label",
    }

    if not required.issubset(predictions.columns):
        return None

    class_dir = output_dir / "classification"
    class_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.to_numeric(
        predictions["truth_class"], errors="coerce"
    ).to_numpy()

    probability = pd.to_numeric(
        predictions["pred_class_probability"], errors="coerce"
    ).to_numpy()

    valid = np.isfinite(truth) & np.isfinite(probability)

    truth = np.rint(truth[valid]).astype(int)
    probability = probability[valid]

    predicted = (probability >= 0.5).astype(int)

    cm = confusion_matrix(truth, predicted, labels=[0, 1])

    metrics = {
        "n": int(len(truth)),
        "accuracy": safe_float(accuracy_score(truth, predicted)),
        "balanced_accuracy": safe_float(
            balanced_accuracy_score(truth, predicted)
        ),
        "precision": safe_float(
            precision_score(truth, predicted, zero_division=0)
        ),
        "recall": safe_float(
            recall_score(truth, predicted, zero_division=0)
        ),
        "f1": safe_float(
            f1_score(truth, predicted, zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "class_report": classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=["RGB (0)", "RC (1)"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of stars")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["RGB", "RC"],
        yticklabels=["RGB", "RC"],
        xlabel="Predicted class",
        ylabel="True class",
        title="Pooled out-of-fold confusion matrix",
    )

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(class_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # ROC curve
    if len(np.unique(truth)) == 2:
        fpr, tpr, thresholds = roc_curve(truth, probability)
        auc = roc_auc_score(truth, probability)
        metrics["roc_auc"] = safe_float(auc)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"ROC (AUC = {auc:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random classifier",
        )

        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Pooled out-of-fold ROC curve")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(class_dir / "roc_curve.png", dpi=180)
        plt.close(fig)

    # Probability distributions by true class
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, name in [(0, "RGB"), (1, "RC")]:
        values = probability[truth == label]

        if len(values):
            ax.hist(
                values,
                bins=30,
                alpha=0.55,
                label=f"{name} (n={len(values)})",
            )

    ax.axvline(0.5, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted probability of RC")
    ax.set_ylabel("Number of stars")
    ax.set_title("Predicted class probabilities")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(class_dir / "class_probabilities.png", dpi=180)
    plt.close(fig)

    save_json(metrics, class_dir / "classification_metrics.json")

    return metrics


# ---------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------

def analyse_training(run_dir, output_dir):
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(run_dir.glob("fold_*"))

    histories = {}
    best_epochs = {}

    fig, ax = plt.subplots(figsize=(9, 6))

    for fold_dir in fold_dirs:
        history_path = fold_dir / "training_history.csv"

        if not history_path.is_file():
            continue

        history = pd.read_csv(history_path)

        if not {"epoch", "train_loss", "val_loss"}.issubset(
            history.columns
        ):
            continue

        fold_name = fold_dir.name
        histories[fold_name] = history

        ax.plot(
            history["epoch"],
            history["train_loss"],
            alpha=0.65,
            linestyle="-",
            label=f"{fold_name} train",
        )

        ax.plot(
            history["epoch"],
            history["val_loss"],
            alpha=0.8,
            linestyle="--",
            label=f"{fold_name} validation",
        )

        best_index = history["val_loss"].idxmin()
        best_epochs[fold_name] = int(
            history.loc[best_index, "epoch"]
        )

    if histories:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss by fold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(training_dir / "loss_curves.png", dpi=180)

    plt.close(fig)

    if best_epochs:
        names = list(best_epochs.keys())
        values = list(best_epochs.values())

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(names, values)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Best epoch")
        ax.set_title("Best epoch per fold")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(training_dir / "best_epochs.png", dpi=180)
        plt.close(fig)

    return {
        "folds_with_history": len(histories),
        "best_epochs": best_epochs,
    }


# ---------------------------------------------------------------------
# Fold comparisons
# ---------------------------------------------------------------------

def analyse_fold_metrics(run_dir, output_dir):
    comparison_dir = output_dir / "fold_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "fold_metrics.csv"

    if not metrics_path.is_file():
        return None

    frame = pd.read_csv(metrics_path)

    excluded = {
        "fold",
        "seed",
        "n_train_samples",
        "n_validation_samples",
        "n_train_kics",
        "n_validation_kics",
        "best_epoch",
        "best_val_loss",
        "final_val_loss",
        "training_time_seconds",
    }

    metric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]

    if not metric_columns:
        return None

    fig, axes = plt.subplots(
        len(metric_columns),
        1,
        figsize=(8, max(4, 3.2 * len(metric_columns))),
        squeeze=False,
    )

    for ax, metric in zip(axes[:, 0], metric_columns):
        values = frame[metric].to_numpy(dtype=float)
        valid = np.isfinite(values)

        ax.bar(
            frame.loc[valid, "fold"].astype(str),
            values[valid],
        )

        mean = np.mean(values[valid]) if valid.any() else np.nan

        if np.isfinite(mean):
            ax.axhline(
                mean,
                linestyle="--",
                label=f"Mean = {mean:.4g}",
            )
            ax.legend()

        ax.set_ylabel(metric)
        ax.set_xlabel("Fold")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Cross-validation metrics by fold")
    fig.tight_layout()
    fig.savefig(comparison_dir / "fold_metrics.png", dpi=180)
    plt.close(fig)

    return {
        "n_folds": int(len(frame)),
        "metrics": metric_columns,
    }


# ---------------------------------------------------------------------
# Optional stellar-parameter analysis
# ---------------------------------------------------------------------

def analyse_stellar_parameters(
    predictions,
    manifest_path,
    targets,
    output_dir,
):
    if manifest_path is None:
        return None

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if "kic" not in manifest.columns:
        raise ValueError("Manifest has no 'kic' column.")

    if "kic" not in predictions.columns:
        return None

    manifest = manifest.copy()
    predictions = predictions.copy()

    manifest["kic"] = manifest["kic"].astype(str)
    predictions["kic"] = predictions["kic"].astype(str)

    # Avoid multiplying prediction rows when the manifest contains
    # several durations for the same KIC.
    metadata = [
        column
        for column in STELLAR_PARAMETERS
        if column in manifest.columns
    ]

    if "duration" in manifest.columns:
        metadata.append("duration")

    metadata = list(dict.fromkeys(metadata))

    if metadata:
        manifest_subset = manifest[["kic"] + metadata].drop_duplicates(
            subset=["kic"]
        )

        merged = predictions.merge(
            manifest_subset,
            on="kic",
            how="left",
            validate="many_to_one",
        )
    else:
        return None

    parameter_dir = output_dir / "stellar_parameters"
    parameter_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for target in targets:
        if target not in REGRESSION_TARGETS:
            continue

        truth_col = f"truth_{target}"
        pred_col = f"pred_{target}"

        if truth_col not in merged or pred_col not in merged:
            continue

        truth = pd.to_numeric(
            merged[truth_col], errors="coerce"
        )
        pred = pd.to_numeric(
            merged[pred_col], errors="coerce"
        )

        residuals = pred - truth

        for parameter in STELLAR_PARAMETERS:
            if parameter not in merged.columns:
                continue

            x = pd.to_numeric(
                merged[parameter], errors="coerce"
            )

            valid = (
                np.isfinite(x)
                & np.isfinite(residuals)
            )

            if valid.sum() < 5:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                x[valid],
                residuals[valid],
                alpha=0.4,
                s=14,
                edgecolors="none",
            )

            ax.axhline(0, linestyle="--", linewidth=1.3)
            ax.set_xlabel(parameter)
            ax.set_ylabel(f"{target} residual (prediction − truth)")
            ax.set_title(f"{target}: residuals vs. {parameter}")
            ax.grid(alpha=0.25)

            fig.tight_layout()

            filename = (
                f"{target}_residuals_vs_{parameter}.png"
            )
            fig.savefig(parameter_dir / filename, dpi=180)
            plt.close(fig)

            results.setdefault(target, []).append(parameter)

    return results


# ---------------------------------------------------------------------
# Pooled metrics and report
# ---------------------------------------------------------------------

def save_pooled_metrics(
    regression_results,
    classification_results,
    output_dir,
):
    pooled = {
        "regression": regression_results,
        "classification": classification_results,
    }

    save_json(pooled, output_dir / "pooled_metrics.json")

    return pooled


def write_readme(
    run_dir,
    output_dir,
    metadata,
    regression_results,
    classification_results,
    fold_info,
):
    lines = [
        "# Cross-validation analysis",
        "",
        f"- Experiment: `{metadata.get('experiment', run_dir.parent.name)}`",
        f"- Model: `{metadata.get('model', 'unknown')}`",
        f"- Input: `{metadata.get('input_mode', 'unknown')}`",
        f"- Duration: `{metadata.get('duration', 'unknown')}`",
        f"- Targets: `{', '.join(metadata.get('targets', []))}`",
        f"- Folds: `{metadata.get('n_folds', 'unknown')}`",
        f"- Seed: `{metadata.get('seed', 'unknown')}`",
        "",
        "## Pooled regression metrics",
        "",
    ]

    for target, values in regression_results.items():
        lines.append(f"### {target}")

        for space, metrics in values.items():
            lines.append(f"**{space}**")

            for metric, value in metrics.items():
                lines.append(f"- {metric}: {value}")

        lines.append("")

    if classification_results:
        lines.extend([
            "## Pooled classification metrics",
            "",
        ])

        for metric, value in classification_results.items():
            if metric != "class_report":
                lines.append(f"- {metric}: {value}")

        lines.append("")

    if fold_info:
        lines.extend([
            "## Fold analysis",
            "",
            f"- Folds in metrics file: {fold_info['n_folds']}",
            f"- Metrics plotted: {', '.join(fold_info['metrics'])}",
            "",
        ])

    lines.extend([
        "## Important interpretation note",
        "",
        "Pooled metrics are calculated from the concatenated held-out "
        "predictions across folds. Fold-level metrics and their "
        "variation are reported separately.",
        "",
        "In the current cross_validate.py implementation, each outer "
        "validation fold is also used for early stopping and scheduler "
        "decisions. Consequently, these metrics are not a fully "
        "independent estimate of generalisation. An inner validation "
        "split within each training fold would be needed for that.",
        "",
        "Regression metrics for nu_max and delta_nu are calculated "
        "both in the model target space (normally log10) and in "
        "physical units after exponentiation.",
        "",
        "The physical-unit conversion assumes that the dataset "
        "target transformation is log10. Confirm this against the "
        "dataset implementation before using the numbers in the thesis.",
        "",
        "## Source",
        "",
        f"CV run: `{run_dir}`",
        "",
    ])

    (output_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def analyse_run(run_dir, output_root, manifest_path=None):
    run_dir = Path(run_dir).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"CV run directory not found: {run_dir}")

    predictions_path = run_dir / "predictions_all_folds.csv"
    metadata_path = run_dir / "metadata.json"

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    metadata = load_json(metadata_path)
    predictions = pd.read_csv(predictions_path)

    if predictions.empty:
        raise ValueError("The predictions file is empty.")

    targets = metadata.get("targets", [])

    if not targets:
        targets = [
            column[len("truth_"):]
            for column in predictions.columns
            if column.startswith("truth_")
        ]

    experiment_name = metadata.get(
        "experiment",
        run_dir.parent.parent.name,
    )

    output_dir = (
        Path(output_root)
        / experiment_name
        / run_dir.name
    ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 72)
    print(f"Run:       {run_dir}")
    print(f"Output:    {output_dir}")
    print(f"Targets:   {targets}")
    print(f"Predictions: {len(predictions)}")

    # Regression
    regression_results = analyse_regression(
        predictions,
        targets,
        output_dir,
    )

    # Classification
    classification_results = None

    if "class" in targets:
        classification_results = analyse_classification(
            predictions,
            output_dir,
        )

    # Training curves
    training_results = analyse_training(
        run_dir,
        output_dir,
    )

    # Fold comparison
    fold_info = analyse_fold_metrics(
        run_dir,
        output_dir,
    )

    # Optional stellar parameter plots
    stellar_results = analyse_stellar_parameters(
        predictions,
        manifest_path,
        targets,
        output_dir,
    )

    # Save pooled metrics
    save_pooled_metrics(
        regression_results,
        classification_results,
        output_dir,
    )

    summary = {
        "source_run": str(run_dir),
        "output_dir": str(output_dir),
        "experiment": experiment_name,
        "model": metadata.get("model"),
        "input_mode": metadata.get("input_mode"),
        "duration": metadata.get("duration"),
        "targets": targets,
        "n_predictions": int(len(predictions)),
        "n_unique_kics": (
            int(predictions["kic"].nunique())
            if "kic" in predictions.columns
            else None
        ),
        "regression": regression_results,
        "classification": classification_results,
        "training": training_results,
        "fold_comparison": fold_info,
        "stellar_parameter_analysis": stellar_results,
    }

    save_json(
        summary,
        output_dir / "analysis_summary.json",
    )

    write_readme(
        run_dir,
        output_dir,
        metadata,
        regression_results,
        classification_results,
        fold_info,
    )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print(f"Summary:          {output_dir / 'analysis_summary.json'}")
    print(f"Report:           {output_dir / 'README.md'}")

    return output_dir


def completed_run_problem(run_dir):
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return "missing metadata.json"

    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return "invalid metadata.json"

    if not isinstance(metadata, dict):
        return "metadata.json is not a JSON object"

    if metadata.get("status") != "completed":
        return f"status is {metadata.get('status', 'unknown')}"

    expected_folds = metadata.get("n_folds")
    completed_folds = metadata.get("n_folds_completed")
    if expected_folds is not None and completed_folds != expected_folds:
        return (
            f"completed {completed_folds} of {expected_folds} folds"
        )

    required_files = (
        "predictions_all_folds.csv",
        "fold_metrics.csv",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        return f"missing {', '.join(missing)}"

    return None


def discover_completed_runs(runs_root):
    runs_root = Path(runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"CV runs root not found: {runs_root}")

    completed = []
    for run_dir in sorted(runs_root.glob("**/cv_runs/*")):
        if not run_dir.is_dir():
            continue

        problem = completed_run_problem(run_dir)
        if problem is None:
            completed.append(run_dir)
        else:
            print(f"Skipping incomplete CV run {run_dir}: {problem}")

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Analyse saved K-fold cross-validation results."
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Analyze one timestamped CV run instead of discovering runs.",
    )

    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Experiment directory to scan for completed cv_runs folders.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for analysis outputs.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest for residuals vs. stellar parameters.",
    )

    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir).expanduser().resolve()]
    else:
        run_dirs = discover_completed_runs(args.runs_root)
        if not run_dirs:
            print(f"No completed CV runs found under {args.runs_root}.")
            return

    for run_dir in run_dirs:
        analyse_run(
            run_dir=run_dir,
            output_root=args.output_root,
            manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()
