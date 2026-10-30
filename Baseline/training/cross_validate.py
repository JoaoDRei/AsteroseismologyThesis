
#!/usr/bin/env python3
"""
K-fold cross-validation for the thesis asteroseismology experiments.

Usage:
    python -m Baseline.training.cross_validate --config Baseline/experiments/01_architecture/A1_resnet1d_lc_80d/config.yaml

Configuration:
    training:
        n_folds: 5
        seed: 42
        epochs: 100
        batch_size: 32
        early_stopping:
            enabled: true
            patience: 15
            min_delta: 0.0001
            restore_best_weights: true

Results are written to a timestamped directory inside the experiment
directory. Each fold holds out complete KICs, so variants of the same
star cannot appear in both its training and validation sets.
"""

import argparse
import copy
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
)

from torch.utils.data import DataLoader, Subset

from Baseline.paths import BASELINE_DIR, PROJECT_ROOT
from Baseline.utils import AstroBaselineDataset, SyntheticPSDPeakDataset

from Baseline.models import (
    SimpleMLP,
    SimpleCNN,
    SimpleTransformer,
    MobileNet1D,
    ResNet1D,
    CNNLSTM,
    AstroConformer,
    AstroConformerV2,
    CNNBiLSTM,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Configuration and model construction
# ---------------------------------------------------------------------

def make_loss(loss_name):
    """Construct a loss function using the names accepted by train.py."""
    name = str(loss_name).lower()

    if name in ("mse", "mse_loss"):
        return nn.MSELoss()

    if name in ("l1", "mae", "l1_loss"):
        return nn.L1Loss()

    if name in ("huber", "smoothl1", "smooth_l1"):
        return nn.SmoothL1Loss()

    if name in ("bce", "bcewithlogits", "bce_with_logits"):
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported loss function: {loss_name!r}")


def build_model(config, input_length, output_dim):
    """Use the same model constructors and configuration keys as train.py."""
    model_config = config["model"]
    model_type = model_config["type"].lower()

    if model_type == "cnn":
        model = SimpleCNN(output_dim=output_dim)

    elif model_type == "mlp":
        model = SimpleMLP(
            input_size=input_length,
            output_dim=output_dim,
        )

    elif model_type == "transformer":
        model = SimpleTransformer(
            seq_length=config["data"]["seq_length"],
            output_dim=output_dim,
        )

    elif model_type == "resnet1d":
        model = ResNet1D(output_dim=output_dim)

    elif model_type == "mobilenet1d":
        model = MobileNet1D(output_dim=output_dim)

    elif model_type == "cnn_lstm":
        model = CNNLSTM(output_dim=output_dim)

    elif model_type == "cnn_bilstm":
        model = CNNBiLSTM(
            output_dim=output_dim,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
        )

    elif model_type == "astroconformer":
        model = AstroConformer(
            encoder_dim=model_config["encoder_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            stride=model_config["patch_size"],
            conv_kernel_size=model_config["conv_kernel_size"],
            dropout=model_config["dropout"],
            output_dim=output_dim,
        )

    elif model_type == "astroconformer_v2":
        model = AstroConformerV2(
            encoder_dim=model_config["encoder_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            patch_size=model_config["patch_size"],
            conv_kernel_size=model_config["conv_kernel_size"],
            dropout=model_config["dropout"],
            output_dim=output_dim,
        )

    else:
        raise ValueError(
            f"Unsupported model type {model_type!r}. "
            "Add its constructor to build_model()."
        )

    return model


def make_loss_configuration(config, targets):
    model_config = config["model"]
    configured_losses = model_config.get("loss_functions", {})
    configured_weights = model_config.get("loss_weights", {})

    losses = {}
    weights = {}

    for target in targets:
        loss_name = configured_losses.get(target, "mse")
        losses[target] = make_loss(loss_name)
        weights[target] = float(configured_weights.get(target, 1.0))

    return losses, weights


def calculate_loss(predictions, targets_tensor, target_names, losses, weights):
    """
    Calculate the weighted multitask loss, matching train.py's convention.
    Predictions and targets have shape [batch, n_targets].
    """
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(-1)

    if targets_tensor.ndim == 1:
        targets_tensor = targets_tensor.unsqueeze(-1)

    if predictions.shape != targets_tensor.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{tuple(predictions.shape)} versus "
            f"{tuple(targets_tensor.shape)}"
        )

    total_loss = torch.zeros((), device=predictions.device)

    for index, target_name in enumerate(target_names):
        pred = predictions[:, index]
        truth = targets_tensor[:, index]

        loss_value = losses[target_name](pred, truth)
        total_loss = total_loss + weights[target_name] * loss_value

    return total_loss


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def normalise_evolution_state(df):
    """Normalize RGB/RC labels as in train.py."""
    df = df.copy()

    if "evolstate" not in df.columns:
        raise ValueError(
            "Classification was requested, but the manifest has no "
            "'evolstate' column."
        )

    labels = df["evolstate"].astype(str).str.strip().str.upper()

    mapping = {
        "0": "RGB",
        "1": "RC",
        "RGB": "RGB",
        "RC": "RC",
    }

    df["evolstate_norm"] = labels.map(mapping)
    df = df[df["evolstate_norm"].isin(["RGB", "RC"])].copy()

    return df


def load_manifest_data(config, targets):
    data_config = config["data"]
    manifest_path = Path(data_config["manifest_path"])

    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    if "kic" not in df.columns:
        raise ValueError("Manifest must contain a 'kic' column.")

    df = df.dropna(subset=["kic"]).copy()

    # Match train.py's class-label preparation.
    if "class" in targets:
        df = normalise_evolution_state(df)

    # Match train.py's duration filter.
    duration = data_config.get("filter_duration")

    if duration and "duration" in df.columns:
        df = df[df["duration"] == duration].copy()

    # Match train.py's metallicity filtering.
    if "metallicity" in targets:
        if "metallicity" not in df.columns:
            raise ValueError(
                "Metallicity is a requested target, but the manifest "
                "has no 'metallicity' column."
            )

        df["metallicity"] = pd.to_numeric(
            df["metallicity"], errors="coerce"
        )
        df = df[
            df["metallicity"].notna()
            & (df["metallicity"] != -9999)
        ].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after applying the manifest filters. "
            "Check the duration, target labels, and manifest path."
        )

    # Apply sample_size to unique stars before making folds.
    sample_size = data_config.get("sample_size")

    if sample_size is not None:
        sample_size = int(sample_size)

        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")

        unique_kics = df["kic"].drop_duplicates().to_numpy()

        if sample_size < len(unique_kics):
            rng = np.random.default_rng(
                int(config["training"].get("seed", 42))
            )
            selected = rng.choice(
                unique_kics,
                size=sample_size,
                replace=False,
            )
            df = df[df["kic"].isin(selected)].copy()

    df = df.reset_index(drop=True)

    if df["kic"].nunique() < 2:
        raise ValueError(
            "At least two unique KICs are required for cross-validation."
        )

    return df


def make_kic_folds(df, config, targets, n_folds, seed):
    """
    Generate folds using unique KICs, then expand each fold to all rows
    belonging to those KICs.
    """
    unique_kics = df["kic"].drop_duplicates().to_numpy()

    if "class" in targets:
        # Each star must have one consistent evolutionary class for
        # stratification. Conflicting labels are an invalid input.
        labels_per_kic = (
            df[["kic", "evolstate_norm"]]
            .drop_duplicates()
        )

        if labels_per_kic["kic"].duplicated().any():
            raise ValueError(
                "At least one KIC has conflicting evolutionary labels."
            )

        labels_per_kic = labels_per_kic.set_index("kic")
        labels = labels_per_kic.loc[
            unique_kics, "evolstate_norm"
        ].to_numpy()

        classes, counts = np.unique(labels, return_counts=True)

        if len(classes) < 2:
            raise ValueError(
                "Stratified classification CV requires both RGB and RC "
                "classes in the filtered dataset."
            )

        if counts.min() < n_folds:
            raise ValueError(
                f"Cannot make {n_folds} stratified folds: the smallest "
                f"class has only {counts.min()} unique KICs. Reduce "
                "training.n_folds or increase the sample."
            )

        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        splits = splitter.split(unique_kics, labels)

    else:
        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        splits = splitter.split(unique_kics)

    folds = []

    for train_star_indices, val_star_indices in splits:
        train_kics = unique_kics[train_star_indices]
        val_kics = unique_kics[val_star_indices]

        train_rows = np.flatnonzero(
            df["kic"].isin(train_kics).to_numpy()
        )
        val_rows = np.flatnonzero(
            df["kic"].isin(val_kics).to_numpy()
        )

        if set(df.iloc[train_rows]["kic"]) & set(df.iloc[val_rows]["kic"]):
            raise RuntimeError("KIC leakage detected between fold splits.")

        folds.append((train_rows, val_rows))

    return folds


# ---------------------------------------------------------------------
# Dataset and loader creation
# ---------------------------------------------------------------------

def make_real_datasets(df, train_rows, val_rows, config, targets):
    """
    Create separate training and validation datasets, matching train.py:
    stochastic training dataset and deterministic validation dataset.
    """
    data_config = config["data"]

    train_df = df.iloc[train_rows].reset_index(drop=True)
    val_df = df.iloc[val_rows].reset_index(drop=True)

    common = {
        "targets": targets,
        "mode": data_config["mode"],
        "target_length": data_config["target_length"],
    }

    train_ds = AstroBaselineDataset(
        train_df,
        **common,
        deterministic=False,
    )

    val_ds = AstroBaselineDataset(
        val_df,
        **common,
        deterministic=True,
    )

    return train_ds, val_ds


def make_synthetic_dataset(config):
    data_config = config["data"]
    target_length = data_config["target_length"]

    return SyntheticPSDPeakDataset(
        n_samples=int(
            config.get("debug", {}).get("synthetic_n_samples", 2000)
        ),
        length=target_length,
    )


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def batch_values_to_list(values):
    """Convert collated KIC values to a flat Python list."""
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()

    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()

    if isinstance(values, (list, tuple)):
        flattened = []

        for value in values:
            if torch.is_tensor(value):
                flattened.extend(
                    value.detach().cpu().reshape(-1).tolist()
                )
            elif isinstance(value, np.ndarray):
                flattened.extend(value.reshape(-1).tolist())
            else:
                flattened.append(value)

        return flattened

    return [values]


# ---------------------------------------------------------------------
# One epoch / validation
# ---------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    device,
    targets,
    losses,
    weights,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    all_truth = []
    all_predictions = []
    all_kics = []

    for x, y, kics in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(x)

            if predictions.ndim == 1:
                predictions = predictions.unsqueeze(-1)

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            loss = calculate_loss(
                predictions,
                y,
                targets,
                losses,
                weights,
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_truth.append(y.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_kics.extend(batch_values_to_list(kics))

    if total_samples == 0:
        raise RuntimeError("Encountered an empty data loader.")

    return {
        "loss": total_loss / total_samples,
        "truth": np.concatenate(all_truth, axis=0),
        "predictions": np.concatenate(all_predictions, axis=0),
        "kics": all_kics,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_fold_metrics(truth, predictions, targets):
    """
    Calculate metrics separately for each target.

    Regression metrics are calculated in the target representation
    returned by AstroBaselineDataset (for nu_max and delta_nu, this is
    normally log10 space in the current pipeline).
    """
    results = {}

    truth = np.asarray(truth)
    predictions = np.asarray(predictions)

    if truth.ndim == 1:
        truth = truth[:, None]

    if predictions.ndim == 1:
        predictions = predictions[:, None]

    for index, target in enumerate(targets):
        y_true = truth[:, index].astype(float)
        y_pred = predictions[:, index].astype(float)

        if target == "class":
            y_true_binary = np.rint(y_true).astype(int)
            y_pred_binary = (
                1.0 / (1.0 + np.exp(-np.clip(y_pred, -60, 60)))
                >= 0.5
            ).astype(int)

            results["class_accuracy"] = float(
                accuracy_score(y_true_binary, y_pred_binary)
            )
            results["class_balanced_accuracy"] = float(
                balanced_accuracy_score(y_true_binary, y_pred_binary)
            )
            results["class_precision"] = float(
                precision_score(
                    y_true_binary,
                    y_pred_binary,
                    zero_division=0,
                )
            )
            results["class_recall"] = float(
                recall_score(
                    y_true_binary,
                    y_pred_binary,
                    zero_division=0,
                )
            )
            results["class_f1"] = float(
                f1_score(
                    y_true_binary,
                    y_pred_binary,
                    zero_division=0,
                )
            )

            cm = confusion_matrix(
                y_true_binary,
                y_pred_binary,
                labels=[0, 1],
            )

            results["class_tn"] = int(cm[0, 0])
            results["class_fp"] = int(cm[0, 1])
            results["class_fn"] = int(cm[1, 0])
            results["class_tp"] = int(cm[1, 1])

        else:
            valid = np.isfinite(y_true) & np.isfinite(y_pred)

            if not valid.any():
                results[f"{target}_mae"] = float("nan")
                results[f"{target}_mse"] = float("nan")
                results[f"{target}_rmse"] = float("nan")
                continue

            y_true = y_true[valid]
            y_pred = y_pred[valid]

            mse = mean_squared_error(y_true, y_pred)

            results[f"{target}_mae"] = float(
                mean_absolute_error(y_true, y_pred)
            )
            results[f"{target}_mse"] = float(mse)
            results[f"{target}_rmse"] = float(np.sqrt(mse))

    return results


def save_fold_predictions(path, kics, truth, predictions, targets):
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)

    if truth.ndim == 1:
        truth = truth[:, None]

    if predictions.ndim == 1:
        predictions = predictions[:, None]

    if len(kics) != len(truth):
        raise ValueError(
            "Number of KIC values does not match prediction count: "
            f"{len(kics)} versus {len(truth)}"
        )

    frame = pd.DataFrame({"kic": kics})

    for index, target in enumerate(targets):
        frame[f"truth_{target}"] = truth[:, index]
        frame[f"pred_{target}"] = predictions[:, index]

        if target == "class":
            probabilities = 1.0 / (
                1.0 + np.exp(
                    -np.clip(predictions[:, index], -60, 60)
                )
            )
            frame["pred_class_probability"] = probabilities
            frame["pred_class_label"] = (
                probabilities >= 0.5
            ).astype(int)

        else:
            frame[f"residual_{target}"] = (
                predictions[:, index] - truth[:, index]
            )

    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Cross-validation orchestration
# ---------------------------------------------------------------------

def cross_validate(config_path):
    start_time = time.time()

    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    training_config = config.setdefault("training", {})
    data_config = config.setdefault("data", {})
    model_config = config["model"]

    seed = int(training_config.get("seed", 42))
    n_folds = int(training_config.get("n_folds", 5))
    epochs = int(
        training_config.get(
            "epochs",
            model_config.get("epochs", 50),
        )
    )
    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    if n_folds < 2:
        raise ValueError("training.n_folds must be at least 2.")

    if epochs < 1:
        raise ValueError("training.epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("training.batch_size must be positive.")

    targets = list(model_config.get("targets", []))

    if not targets:
        raise ValueError(
            "model.targets is empty. Cross-validation requires at least "
            "one supervised target; the autoencoder uses a separate trainer."
        )

    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        device = torch.device("cpu")

    set_seed(seed)

    # Output path: unique directory per CV execution.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_dir = config_path.parent
    output_dir = experiment_dir / "cv_runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata = {
        "experiment": experiment_dir.name,
        "config_path": str(config_path),
        "model": model_config["type"],
        "input_mode": data_config.get("mode"),
        "duration": data_config.get("filter_duration"),
        "targets": targets,
        "seed": seed,
        "n_folds": n_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "started_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    losses, weights = make_loss_configuration(config, targets)

    # Early stopping settings mirror train.py.
    early_config = training_config.get("early_stopping", {})

    if isinstance(early_config, bool):
        early_config = {"enabled": early_config}

    early_enabled = early_config.get("enabled", True)
    patience = int(early_config.get("patience", 15))
    min_delta = float(early_config.get("min_delta", 1e-4))
    restore_best = early_config.get("restore_best_weights", True)

    learning_rate = float(
        training_config.get(
            "learning_rate",
            model_config.get("learning_rate", 1e-3),
        )
    )

    scheduler_patience = int(
        training_config.get("scheduler_patience", 5)
    )

    synthetic = bool(
        config.get("debug", {}).get("synthetic", False)
    )

    if synthetic:
        dataset = make_synthetic_dataset(config)
        all_indices = np.arange(len(dataset))

        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )

        folds = list(splitter.split(all_indices))
        df = None

    else:
        df = load_manifest_data(config, targets)

        folds = make_kic_folds(
            df=df,
            config=config,
            targets=targets,
            n_folds=n_folds,
            seed=seed,
        )

        dataset = None

    fold_metrics = []
    fold_prediction_paths = []

    print("=" * 72)
    print("K-FOLD CROSS-VALIDATION")
    print("=" * 72)
    print(f"Config:       {config_path}")
    print(f"Output:       {output_dir}")
    print(f"Model:        {model_config['type']}")
    print(f"Input:        {data_config.get('mode')}")
    print(f"Duration:     {data_config.get('filter_duration')}")
    print(f"Targets:      {targets}")
    print(f"Folds:        {n_folds}")
    print(f"Seed:         {seed}")
    print(f"Device:       {device}")

    if not synthetic:
        print(f"Unique KICs:  {df['kic'].nunique()}")
        print(f"Manifest rows:{len(df):>7}")

    for fold_number, (train_indices, val_indices) in enumerate(
        folds,
        start=1,
    ):
        fold_start = time.time()
        fold_seed = seed + fold_number
        set_seed(fold_seed)

        fold_dir = output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=False)

        print()
        print("-" * 72)
        print(f"Fold {fold_number}/{n_folds}")
        print("-" * 72)

        if synthetic:
            train_ds = Subset(dataset, train_indices.tolist())
            val_ds = Subset(dataset, val_indices.tolist())
        else:
            train_ds, val_ds = make_real_datasets(
                df,
                train_indices,
                val_indices,
                config,
                targets,
            )

            train_kics = set(
                df.iloc[train_indices]["kic"].tolist()
            )
            val_kics = set(
                df.iloc[val_indices]["kic"].tolist()
            )

            if train_kics & val_kics:
                raise RuntimeError(
                    f"KIC leakage detected in fold {fold_number}."
                )

            print(f"Training KICs:   {len(train_kics)}")
            print(f"Validation KICs: {len(val_kics)}")

        train_loader = make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        input_length = getattr(
            train_ds,
            "target_length",
            data_config.get("target_length"),
        )

        if input_length is None:
            input_length = data_config.get("target_length")

        model = build_model(
            config,
            input_length=int(input_length),
            output_dim=len(targets),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        best_checkpoint = fold_dir / "best_model.pth"

        for epoch in range(1, epochs + 1):
            train_result = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                targets=targets,
                losses=losses,
                weights=weights,
                optimizer=optimizer,
            )

            val_result = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                targets=targets,
                losses=losses,
                weights=weights,
                optimizer=None,
            )

            train_loss = train_result["loss"]
            val_loss = val_result["loss"]

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })

            print(
                f"Fold {fold_number} | "
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"lr={current_lr:.3e}",
                flush=True,
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save a portable CPU state dict.
                state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(state, best_checkpoint)

            else:
                epochs_without_improvement += 1

            if early_enabled and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch was {best_epoch}.",
                    flush=True,
                )
                break

        pd.DataFrame(history).to_csv(
            fold_dir / "training_history.csv",
            index=False,
        )

        if best_epoch == 0:
            raise RuntimeError(
                f"No best checkpoint was saved for fold {fold_number}."
            )

        if restore_best:
            state = torch.load(
                best_checkpoint,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state)

        # Evaluate the best model on this fold's held-out data.
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            targets=targets,
            losses=losses,
            weights=weights,
            optimizer=None,
        )

        metrics = calculate_fold_metrics(
            val_result["truth"],
            val_result["predictions"],
            targets,
        )

        fold_record = {
            "fold": fold_number,
            "seed": fold_seed,
            "n_train_samples": len(train_ds),
            "n_validation_samples": len(val_ds),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_loss": val_result["loss"],
            "training_time_seconds": time.time() - fold_start,
        }

        if not synthetic:
            fold_record["n_train_kics"] = len(train_kics)
            fold_record["n_validation_kics"] = len(val_kics)

        fold_record.update(metrics)
        fold_metrics.append(fold_record)

        prediction_path = fold_dir / "predictions.csv"

        if synthetic:
            prediction_kics = val_result["kics"]
        else:
            prediction_kics = val_result["kics"]

        save_fold_predictions(
            prediction_path,
            prediction_kics,
            val_result["truth"],
            val_result["predictions"],
            targets,
        )

        fold_prediction_paths.append(prediction_path)

        with (fold_dir / "metrics.json").open("w") as f:
            json.dump(fold_record, f, indent=2)

        print(f"Best epoch:      {best_epoch}")
        print(f"Best val loss:   {best_val_loss:.6f}")
        print(f"Fold metrics:    {fold_dir / 'metrics.json'}")

        del model, optimizer, scheduler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate fold metrics
    # -----------------------------------------------------------------

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    excluded = {
        "fold",
        "seed",
        "n_train_samples",
        "n_validation_samples",
        "n_train_kics",
        "n_validation_kics",
        "best_epoch",
        "training_time_seconds",
    }

    metric_columns = [
        column
        for column in fold_df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]

    summary_records = []

    for metric in metric_columns:
        values = pd.to_numeric(
            fold_df[metric],
            errors="coerce",
        ).to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        summary_records.append({
            "metric": metric,
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values, ddof=1))
            if len(values) > 1 else 0.0 if len(values) == 1 else np.nan,
            "n_folds": int(len(values)),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    # Combine held-out predictions. Every row was predicted by a model
    # that did not train on that row's KIC.
    prediction_frames = []

    for fold_number, prediction_path in enumerate(
        fold_prediction_paths,
        start=1,
    ):
        frame = pd.read_csv(prediction_path)
        frame.insert(0, "fold", fold_number)
        prediction_frames.append(frame)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    all_predictions.to_csv(
        output_dir / "predictions_all_folds.csv",
        index=False,
    )

    metadata.update({
        "finished_at": datetime.now().isoformat(),
        "training_time_seconds": time.time() - start_time,
        "status": "completed",
        "n_folds_completed": len(fold_metrics),
    })

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Plain-text summary for quick inspection.
    with (output_dir / "summary.txt").open("w") as f:
        f.write("K-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Experiment: {experiment_dir.name}\n")
        f.write(f"Model: {model_config['type']}\n")
        f.write(f"Input: {data_config.get('mode')}\n")
        f.write(f"Duration: {data_config.get('filter_duration')}\n")
        f.write(f"Targets: {', '.join(targets)}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Device: {device}\n\n")
        f.write("METRICS (mean and standard deviation across folds)\n")
        f.write("-" * 72 + "\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['metric']:<32} "
                f"{row['mean']:.6f} +/- {row['std']:.6f} "
                f"(n={int(row['n_folds'])})\n"
            )

        f.write("\n")
        f.write(f"Results directory: {output_dir}\n")

    print()
    print("=" * 72)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 72)
    print(f"Fold metrics:    {output_dir / 'fold_metrics.csv'}")
    print(f"Summary metrics: {output_dir / 'summary_metrics.csv'}")
    print(f"All predictions: {output_dir / 'predictions_all_folds.csv'}")
    print(f"Text summary:    {output_dir / 'summary.txt'}")
    print(f"Output directory:{output_dir}")

    if not summary_df.empty:
        print("\nMean +/- standard deviation:")
        for _, row in summary_df.iterrows():
            print(
                f"  {row['metric']}: "
                f"{row['mean']:.6f} +/- {row['std']:.6f}"
            )

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run K-fold cross-validation for an experiment config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASELINE_DIR / "config.yaml"),
        help="Path to the experiment YAML configuration.",
    )

    args = parser.parse_args()
    cross_validate(args.config)


if __name__ == "__main__":
    main()
