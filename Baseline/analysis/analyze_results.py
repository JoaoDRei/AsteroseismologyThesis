
#!/usr/bin/env python3
"""
Analyze saved asteroseismology ML experiments.

Outputs:
    Baseline/results_analysis/all_runs.csv
    Baseline/results_analysis/selected_runs.csv
    Baseline/results_analysis/plots/*.png

All runs are preserved in all_runs.csv. For the initial comparison,
the latest complete run is selected for each model/input/duration/target
configuration. Runs with different configurations are never merged.
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from Baseline.paths import BASELINE_DIR


RESULTS = BASELINE_DIR / "results"
OUT = BASELINE_DIR / "results_analysis"
PLOTS = OUT / "plots"

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def read_json(path):
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def read_config(path):
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get_nested(d, *keys, default=None):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


# REGRESSION METRICS
#
# RMSE (Root Mean Squared Error):
#   Square root of the average squared prediction error.
#   Lower is better; 0 means perfect predictions.
#   Large errors are penalized more heavily because errors are squared.
#   Expressed in the same units as the target, unless the target was
#   transformed (e.g. log10(nu_max)) before calculating the metric.
#
# MSE (Mean Squared Error):
#   Average squared difference between predictions and true values.
#   Lower is better; 0 means perfect predictions.
#   More sensitive to large errors than MAE.
#   Units are the square of the target's units.
#
# MAE (Mean Absolute Error):
#   Average absolute difference between predictions and true values.
#   Lower is better; 0 means perfect predictions.
#   Easier to interpret as a typical absolute prediction error.
#   Less sensitive to a few large errors than RMSE.
#
# IMPORTANT:
#   Check whether the training/evaluation pipeline computes these
#   metrics in the original target units or in transformed units.
#   If y = log10(nu_max) is used for evaluation, RMSE/MAE describe
#   errors in log10(nu_max), not directly in microhertz.

def load_runs():
    records = []

    for folder in sorted(RESULTS.iterdir()):
        if not folder.is_dir():
            continue

        metadata_path = folder / "metadata.json"
        metrics_path = folder / "metrics.json"
        config_path = folder / "config.yaml"

        # Ignore folders that contain no usable experiment metadata.
        if not metadata_path.exists() and not metrics_path.exists():
            continue

        meta = read_json(metadata_path)
        metrics = read_json(metrics_path)
        config = read_config(config_path)

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        training_cfg = config.get("training", {})

        model = meta.get("model", model_cfg.get("type", "unknown"))
        mode = meta.get("input_mode", data_cfg.get("mode", "unknown"))
        duration = meta.get(
            "duration", data_cfg.get("filter_duration", "unknown")
        )

        targets = meta.get("targets", model_cfg.get("targets", []))
        if isinstance(targets, str):
            targets = [targets]

        # Required metrics for determining whether a run is complete.
        expected = []
        for target in targets:
            if target in ("nu_max", "delta_nu"):
                expected.append(f"{target}_rmse")
            elif target == "class":
                expected.append("class_balanced_accuracy")
            elif target == "metallicity":
                expected.append("metallicity_rmse")

        missing = [key for key in expected if key not in metrics]
        complete = (
            metadata_path.exists()
            and metrics_path.exists()
            and bool(expected)
            and not missing
        )

        record = {
            "run_dir": str(folder),
            "experiment_name": meta.get("experiment_name", folder.name),
            "timestamp": meta.get("timestamp", folder.name),
            "model": model,
            "input": mode,
            "duration": duration,
            "targets": "+".join(sorted(targets)),
            "complete": complete,
            "missing_metrics": ";".join(missing),
            "seed": meta.get("seed"),
            "device": meta.get("device"),
            "parameters": meta.get("trainable_parameters"),
            "training_time_s": meta.get("training_time_seconds"),
            "best_epoch": meta.get("best_epoch"),
            "best_validation_loss": meta.get("best_validation_loss"),
            "test_loss": metrics.get("test_loss"),
            "nu_max_rmse": metrics.get("nu_max_rmse"),
            "nu_max_mse": metrics.get("nu_max_mse"),
            "delta_nu_rmse": metrics.get("delta_nu_rmse"),
            "delta_nu_mse": metrics.get("delta_nu_mse"),
            "metallicity_rmse": metrics.get("metallicity_rmse"),
            "metallicity_mae": metrics.get("metallicity_mae"),
            "class_accuracy": metrics.get("class_accuracy"),
            "class_balanced_accuracy": metrics.get(
                "class_balanced_accuracy"
            ),
            "class_precision": metrics.get("class_precision"),
            "class_recall": metrics.get("class_recall"),
            "class_f1": metrics.get("class_f1"),
        }

        # Preserve other metrics rather than discarding them.
        for key, value in metrics.items():
            if key not in record:
                record[key] = value

        records.append(record)

    return pd.DataFrame(records)

# LOSS METRICS
#
# Training loss:
#   The objective the model minimizes using the training data.
#   A decreasing training loss generally means the model is fitting
#   the training examples more closely.
#
# Validation loss:
#   Loss calculated on held-out validation data during training.
#   Used to monitor generalization and select the best model/checkpoint.
#   A validation loss that stops improving while training loss keeps
#   decreasing can indicate overfitting.
#
# Test loss:
#   Loss calculated on the held-out test set after training.
#   Estimates performance on unseen data, provided the test set was
#   not used to tune the model or select hyperparameters.
#
# Lower is generally better, but the interpretation depends on the
# loss function (e.g. MSE, MAE or a weighted multitask combination).
# For multitask models, total loss may combine several targets, so
# it is not necessarily directly comparable across different target
# combinations or differently weighted objectives.

def select_latest_complete(df):
    """Select latest complete run per configuration, preserving all runs."""
    valid = df[df["complete"]].copy()

    if valid.empty:
        return valid

    # Configuration identity includes targets: classification-only and
    # multitask experiments must not be treated as the same configuration.
    group_cols = ["model", "input", "duration", "targets", "seed"]

    valid = valid.sort_values("timestamp")
    return valid.drop_duplicates(group_cols, keep="last").reset_index(drop=True)


def save_table(df, filename):
    df.to_csv(OUT / filename, index=False)
    print(f"Saved {OUT / filename}")


def bar_plot(df, metric, title, filename, subset=None):
    data = df.copy()

    if subset is not None:
        if callable(subset):
            data = data[subset(data)]
        else:
            data = data[subset]

    data = data.dropna(subset=[metric])
    if data.empty:
        return

    # Avoid mixing different targets or durations in one leaderboard.
    labels = (
        data["model"].astype(str)
        + " | "
        + data["input"].astype(str)
        + " | "
        + data["duration"].astype(str)
    )
    data = data.assign(label=labels).sort_values(metric)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(data))))
    ax.barh(data["label"], data[metric])
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=180)
    plt.close(fig)
    print(f"Saved {PLOTS / filename}")


def main():
    df = load_runs()

    if df.empty:
        print(f"No experiment results found under {RESULTS}")
        return

    save_table(df, "all_runs.csv")

    print(f"\nResult folders found: {len(df)}")
    print(f"Complete runs: {int(df['complete'].sum())}")
    print(f"Incomplete runs: {int((~df['complete']).sum())}")

    if (~df["complete"]).any():
        print("\nIncomplete runs:")
        print(df.loc[~df["complete"],
                     ["experiment_name", "missing_metrics"]].to_string(
                         index=False
                     ))

    selected = select_latest_complete(df)
    save_table(selected, "selected_runs.csv")

    if selected.empty:
        print("No complete runs with the expected metrics were found.")
        return

    # A. Architecture comparisons: common LC, 80d, multitask experiments.

    # ARCHITECTURE COMPARISON
    # Compares model architectures using the same input representation,
    # observation duration and target combination. Lower regression RMSE
    # is better; higher balanced accuracy is better.
    # Keeping these conditions fixed makes architecture the main intended
    # experimental difference.
    common = (
        (selected["input"] == "lc")
        & (selected["duration"] == "80d")
        & (selected["targets"] == "class+delta_nu+nu_max")
    )

    bar_plot(
        selected, "nu_max_rmse",
        "Architecture comparison: nu_max RMSE (LC, 80d)",
        "architecture_nu_max_rmse.png", common
    )
    bar_plot(
        selected, "delta_nu_rmse",
        "Architecture comparison: delta_nu RMSE (LC, 80d)",
        "architecture_delta_nu_rmse.png", common
    )
    bar_plot(
        selected, "class_balanced_accuracy",
        "Architecture comparison: balanced accuracy (LC, 80d)",
        "architecture_class_balanced_accuracy.png", common
    )

    # B. LC versus PSD: paired architectures with the same targets/duration.
    
    # LC VERSUS PSD
    # Compares light-curve and power-spectrum inputs for the same model,
    # duration and targets. This tests whether temporal information in the
    # light curve or frequency-domain information in the PSD is more useful.
    # A fair comparison also requires consistent data splits, target scaling,
    # evaluation procedures and comparable training settings.
    for metric, label in [
        ("nu_max_rmse", "nu_max RMSE"),
        ("delta_nu_rmse", "delta_nu RMSE"),
        ("class_balanced_accuracy", "Class balanced accuracy"),
    ]:
        subset = selected[
            (selected["duration"] == "80d")
            & (selected["targets"] == "class+delta_nu+nu_max")
            & selected["input"].isin(["lc", "psd"])
        ].dropna(subset=[metric])

        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="model", columns="input", values=metric, aggfunc="first"
        )
        pivot = pivot.dropna(subset=["lc", "psd"], how="any")
        if pivot.empty:
            continue

        ax = pivot[["lc", "psd"]].plot(
            kind="bar", figsize=(9, 5), rot=30
        )
        ax.set_ylabel(label)
        ax.set_title(f"LC versus PSD: {label}")
        ax.figure.tight_layout()
        ax.figure.savefig(
            PLOTS / f"lc_vs_psd_{metric}.png", dpi=180
        )
        plt.close(ax.figure)

    # C. Duration comparison: same model/input/targets, different durations.
    # OBSERVATION-DURATION COMPARISON
    # Compares performance for 20-, 55- and 80-day observations.
    # This tests how the amount of observed stellar variability affects
    # prediction quality. Lower regression RMSE and higher balanced accuracy
    # indicate better performance.
    duration_data = selected[
        selected["input"].eq("lc")
        & selected["duration"].isin(["20d", "55d", "80d"])
        & selected["targets"].eq("class+delta_nu+nu_max")
    ]

    for metric, label in [
        ("nu_max_rmse", "nu_max RMSE"),
        ("delta_nu_rmse", "delta_nu RMSE"),
        ("class_balanced_accuracy", "Class balanced accuracy"),
    ]:
        data = duration_data.dropna(subset=[metric])
        for model, group in data.groupby("model"):
            group = group.copy()
            order = {"20d": 20, "55d": 55, "80d": 80}
            group["days"] = group["duration"].map(order)
            group = group.sort_values("days")
            if len(group) < 2:
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(group["days"], group[metric], marker="o")
            ax.set_xlabel("Observation duration (days)")
            ax.set_ylabel(label)
            ax.set_title(f"{model}: {label} versus duration")
            ax.set_xticks([20, 55, 80])
            fig.tight_layout()
            fig.savefig(
                PLOTS / f"duration_{model}_{metric}.png", dpi=180
            )
            plt.close(fig)


    # CLASSIFICATION METRICS
    #
    # Accuracy:
    #   Fraction of all predictions that are correct.
    #   Higher is better; 1.0 means every prediction is correct.
    #   Can be misleading when one class is much more common than another.
    #
    # Balanced accuracy:
    #   Average recall across all classes.
    #   Gives each class equal weight, making it useful when the dataset
    #   contains unequal numbers of RGB and RC stars.
    #   Higher is better; 1.0 means perfect classification.
    #
    # Precision:
    #   Of the stars predicted to belong to a given class, the fraction
    #   that actually belong to that class.
    #   Higher precision means fewer false-positive predictions.
    #
    # Recall:
    #   Of the stars that truly belong to a given class, the fraction
    #   correctly identified by the model.
    #   Higher recall means fewer missed members of that class.
    #
    # F1 score:
    #   Harmonic mean of precision and recall:
    #       F1 = 2 * precision * recall / (precision + recall)
    #   Balances precision and recall; it is high only when both are high.
    #   Higher is better; 1.0 means perfect precision and recall.
    #   Unlike accuracy, F1 does not account for true negatives directly.
    #
    # NOTE:
    #   Confirm whether precision, recall and F1 are binary, macro-averaged,
    #   weighted-averaged or calculated for a particular class.
    #   Macro-F1 averages the F1 score of each class equally, while
    #   weighted-F1 weights each class according to its sample count.



    # D. Classification metrics for all selected classification runs.
    bar_plot(
        selected, "class_f1",
        "Classification F1 score",
        "classification_f1.png",
        lambda d: d["targets"].str.contains("class", na=False)
    )


        # E. Metallicity regression experiments.
    # Compare only metallicity-target experiments; do not mix them with
    # nu_max / Delta_nu regression or classification experiments.

    metallicity_data = selected[
        selected["targets"].eq("metallicity")
    ].copy()

        # MODEL COMPLEXITY AND COMPUTATIONAL COST
        #
        # Trainable parameters:
        #   Number of model parameters updated during training.
        #   More parameters generally mean greater model capacity, but do
        #   not guarantee better predictions.
        #   Larger models can require more memory and be more prone to
        #   overfitting when training data are limited.
        #
        # Training time:
        #   Time taken to train the model for the recorded experiment.
        #   Lower is faster and can make experimentation more practical.
        #   Depends on hardware, data loading, batch size, number of epochs,
        #   and early stopping, so compare times under similar conditions.
        #
        # Best epoch:
        #   Epoch associated with the selected best validation checkpoint
        #   (assuming the training pipeline records it that way).
        #   Reaching a best epoch early is not automatically better; the
        #   validation-loss curve and stopping criteria also matter.
        #
        # Best validation loss:
        #   Lowest recorded validation objective, if that is how the
        #   training pipeline defines and saves the best checkpoint.
        #   Lower is generally better only when the loss definitions and
        #   target combinations are comparable.

    if not metallicity_data.empty:
        bar_plot(
            metallicity_data,
            "metallicity_rmse",
            "Metallicity regression: RMSE",
            "metallicity_rmse.png",
        )

        bar_plot(
            metallicity_data,
            "metallicity_mae",
            "Metallicity regression: MAE",
            "metallicity_mae.png",
        )

        bar_plot(
            metallicity_data,
            "training_time_s",
            "Metallicity regression: training time",
            "metallicity_training_time.png",
        )
    else:
        print("No selected metallicity experiments found.")






    print(f"\nAnalysis complete. Tables: {OUT}")
    print(f"Plots: {PLOTS}")

    
    # E. Two-target regression: nu_max + delta_nu, without classification.
    # Keep these results separate from the three-target multitask experiments.

    two_target = (
        (selected["duration"] == "80d")
        & (selected["targets"] == "delta_nu+nu_max")
    )

    two_target_lc = two_target & (selected["input"] == "lc")

    bar_plot(
        selected,
        "nu_max_rmse",
        "Two-target architecture comparison: nu_max RMSE (LC, 80d)",
        "two_target_architecture_nu_max_rmse.png",
        two_target_lc,
    )

    bar_plot(
        selected,
        "delta_nu_rmse",
        "Two-target architecture comparison: delta_nu RMSE (LC, 80d)",
        "two_target_architecture_delta_nu_rmse.png",
        two_target_lc,
    )

    # LC versus PSD for the two-target experiments.
    
    # TWO-TARGET VERSUS THREE-TARGET COMPARISON
    # Compares seismic regression with and without evolutionary-state
    # classification as an auxiliary task. This tests whether multitask
    # learning is associated with improved seismic predictions.
    # Interpret differences cautiously if training settings, optimization
    # or model selection differ between the experiments.
    for metric, label in [
        ("nu_max_rmse", "nu_max RMSE"),
        ("delta_nu_rmse", "delta_nu RMSE"),
    ]:
        subset = selected[
            (selected["duration"] == "80d")
            & (selected["targets"] == "delta_nu+nu_max")
            & selected["input"].isin(["lc", "psd"])
        ].dropna(subset=[metric])

        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="model",
            columns="input",
            values=metric,
            aggfunc="first",
        )

        # Only compare architectures that have both LC and PSD results.
        pivot = pivot.dropna(subset=["lc", "psd"], how="any")

        if pivot.empty:
            print(f"No paired LC/PSD results for two-target {metric}.")
            continue

        ax = pivot[["lc", "psd"]].plot(
            kind="bar",
            figsize=(9, 5),
            rot=30,
        )
        ax.set_ylabel(label)
        ax.set_title(f"Two-target LC versus PSD: {label}")
        ax.figure.tight_layout()
        ax.figure.savefig(
            PLOTS / f"two_target_lc_vs_psd_{metric}.png",
            dpi=180,
        )
        plt.close(ax.figure)

    



if __name__ == "__main__":
    main()





