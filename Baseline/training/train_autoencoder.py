
"""
Train and evaluate a CNN-BiLSTM autoencoder.

Expected interfaces
-------------------
AstroReconstructionDataset(df, target_length=..., reconstruction_length=...,
                           deterministic=...)
    returns (input, reconstruction_target, kic)

CNNBiLSTMAutoencoder(hidden_size=..., num_layers=...,
                     reconstruction_length=...)
    returns (reconstruction, embedding)

Outputs
-------
A unique experiment directory containing:
    config.yaml
    metadata.json
    split_kics.csv
    training_history.csv
    best_model.pth
    metrics.json
    metrics.txt
    per_star_test_metrics.csv
    reconstruction_examples/
"""

import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from Baseline.models import CNNBiLSTMAutoencoder
from Baseline.paths import BASELINE_DIR, PROJECT_ROOT
from Baseline.utils.reconstruction_dataset import AstroReconstructionDataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Give each DataLoader worker a reproducible random seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Configuration and paths
# ============================================================

def load_config(config_path):
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    config["_config_path"] = str(config_path)
    return config


def resolve_path(path, config_path):
    """Resolve a path relative to the working directory or config file."""
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    config_candidate = (
        Path(config_path).parent / path
    ).resolve()
    if config_candidate.exists():
        return config_candidate

    return (PROJECT_ROOT / path).resolve()


def get_settings(config):
    acfg = config.get("autoencoder", {})
    dcfg = config.get("data", {})
    tcfg = config.get("training", {})

    seed = int(tcfg.get("seed", 42))

    # Prefer the autoencoder-specific length. Otherwise reuse the
    # supervised data length, then fall back to the original 65000.
    input_length = int(
        acfg.get(
            "input_length",
            dcfg.get("target_length", 65000)
        )
    )

    reconstruction_length = int(
        acfg["reconstruction_length"]
    )

    batch_size = int(acfg.get("batch_size", dcfg.get("batch_size", 16)))
    num_workers = int(acfg.get("num_workers", dcfg.get("num_workers", 0)))
    epochs = int(acfg.get("epochs", tcfg.get("epochs", 100)))
    learning_rate = float(acfg.get("learning_rate", 1e-3))
    weight_decay = float(acfg.get("weight_decay", 0.0))

    early_cfg = acfg.get("early_stopping", {})
    patience = int(early_cfg.get("patience", 15))
    min_delta = float(early_cfg.get("min_delta", 1e-4))

    scheduler_cfg = acfg.get("scheduler", {})
    scheduler_factor = float(scheduler_cfg.get("factor", 0.5))
    scheduler_patience = int(scheduler_cfg.get("patience", 5))
    scheduler_min_lr = float(scheduler_cfg.get("min_lr", 1e-7))

    split_cfg = acfg.get("split", {})
    train_fraction = float(split_cfg.get("train", 0.8))
    val_fraction = float(split_cfg.get("validation", 0.1))
    test_fraction = float(split_cfg.get("test", 0.1))

    fractions = [train_fraction, val_fraction, test_fraction]
    if any(f <= 0 or f >= 1 for f in fractions):
        raise ValueError("Train, validation and test fractions must be between 0 and 1.")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation and test fractions must sum to 1.")

    if input_length <= 0 or reconstruction_length <= 0:
        raise ValueError("Input and reconstruction lengths must be positive.")

    if batch_size <= 0 or epochs <= 0:
        raise ValueError("Batch size and epoch count must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    return {
        "seed": seed,
        "input_length": input_length,
        "reconstruction_length": reconstruction_length,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "min_delta": min_delta,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "scheduler_min_lr": scheduler_min_lr,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "hidden_size": int(acfg["hidden_size"]),
        "num_layers": int(acfg["num_layers"]),
        "checkpoint_name": acfg.get("checkpoint_name", "best_model.pth"),
        "output_dir": acfg.get(
            "output_dir",
            str(BASELINE_DIR / "results" / "autoencoder")
        ),
        "num_examples": int(acfg.get("num_reconstruction_examples", 8)),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_manifest(config, settings):
    manifest_path = resolve_path(
        config["data"]["manifest_path"],
        config["_config_path"]
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)
    initial_rows = len(df)

    if "kic" not in df.columns:
        raise ValueError("Manifest must contain a 'kic' column.")

    if df["kic"].isna().any():
        missing = int(df["kic"].isna().sum())
        print(f"Removing {missing} rows with missing KIC.", flush=True)
        df = df[df["kic"].notna()].copy()

    # Apply the same duration filter configured for the supervised run.
    duration = config.get("data", {}).get("filter_duration")

    if duration is not None:
        if "duration" not in df.columns:
            raise ValueError(
                "filter_duration is configured, but the manifest "
                "has no 'duration' column."
            )

        before = len(df)
        df = df[df["duration"] == duration].copy()
        print(
            f"Duration filter ({duration}): "
            f"{before} -> {len(df)} rows",
            flush=True,
        )

    # Make KIC identifiers numeric and consistent where possible.
    numeric_kic = pd.to_numeric(df["kic"], errors="coerce")
    invalid_kic = numeric_kic.isna()

    if invalid_kic.any():
        raise ValueError(
            f"{int(invalid_kic.sum())} rows have non-numeric KIC identifiers."
        )

    df["kic"] = numeric_kic.astype("int64")

    duplicate_rows = int(df["kic"].duplicated(keep="first").sum())
    if duplicate_rows:
        print(
            f"Warning: {duplicate_rows} duplicate KIC rows remain "
            "after filtering. Keeping the first row per KIC."
        )
        df = df.drop_duplicates(subset=["kic"], keep="first").copy()

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No stars remain after manifest filtering.")

    if df["kic"].nunique() < 3:
        raise ValueError(
            "At least three unique stars are required for train/validation/test splits."
        )

    print(f"Manifest rows before filtering: {initial_rows}", flush=True)
    print(f"Stars available for reconstruction: {len(df)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    return df, str(manifest_path)


def split_by_kic(df, settings):
    """Split unique stars, then construct row partitions by KIC."""
    kics = np.asarray(sorted(df["kic"].unique()))
    seed = settings["seed"]

    train_fraction = settings["train_fraction"]
    val_fraction = settings["val_fraction"]
    test_fraction = settings["test_fraction"]

    train_kics, remainder_kics = train_test_split(
        kics,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True
    )

    remainder_total = val_fraction + test_fraction
    val_relative = val_fraction / remainder_total

    val_kics, test_kics = train_test_split(
        remainder_kics,
        train_size=val_relative,
        random_state=seed + 1,
        shuffle=True
    )

    partitions = {
        "train": np.asarray(train_kics, dtype=np.int64),
        "validation": np.asarray(val_kics, dtype=np.int64),
        "test": np.asarray(test_kics, dtype=np.int64),
    }

    sets = {name: set(values.tolist()) for name, values in partitions.items()}

    if sets["train"] & sets["validation"]:
        raise RuntimeError("KIC leakage between training and validation sets.")
    if sets["train"] & sets["test"]:
        raise RuntimeError("KIC leakage between training and test sets.")
    if sets["validation"] & sets["test"]:
        raise RuntimeError("KIC leakage between validation and test sets.")

    row_partitions = {
        name: df[df["kic"].isin(kics)].reset_index(drop=True)
        for name, kics in partitions.items()
    }

    for name, part_df in row_partitions.items():
        if part_df.empty:
            raise ValueError(f"The {name} partition is empty.")

    print("\nKIC-level data split:", flush=True)
    for name, part_df in row_partitions.items():
        print(
            f"  {name:10s}: {len(part_df):6d} stars "
            f"({len(part_df) / len(df):.1%})",
            flush=True,
        )

    return partitions, row_partitions


def make_dataset(df, settings, deterministic):
    return AstroReconstructionDataset(
        df,
        target_length=settings["input_length"],
        reconstruction_length=settings["reconstruction_length"],
        deterministic=deterministic
    )


def make_loader(dataset, settings, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=settings["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False
    )


# ============================================================
# Model and tensor validation
# ============================================================

def build_model(settings, device):
    model = CNNBiLSTMAutoencoder(
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        reconstruction_length=settings["reconstruction_length"]
    )

    return model.to(device)


def unpack_batch(batch, device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError(
            "AstroReconstructionDataset must return "
            "(input, target, kic)."
        )

    x, target, kics = batch[:3]

    x = x.to(device, dtype=torch.float32, non_blocking=True)
    target = target.to(device, dtype=torch.float32, non_blocking=True)

    if not torch.isfinite(x).all():
        raise ValueError("Input batch contains NaN or infinite values.")

    if not torch.isfinite(target).all():
        raise ValueError("Reconstruction target contains NaN or infinite values.")

    return x, target, kics


def forward_model(model, x, target):
    result = model(x)

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError(
            "CNNBiLSTMAutoencoder must return "
            "(reconstruction, embedding)."
        )

    reconstruction, embedding = result

    if reconstruction.shape != target.shape:
        raise ValueError(
            "Reconstruction/target shape mismatch: "
            f"output={tuple(reconstruction.shape)}, "
            f"target={tuple(target.shape)}. "
            "Check AstroReconstructionDataset and "
            "CNNBiLSTMAutoencoder output dimensions."
        )

    if not torch.isfinite(reconstruction).all():
        raise ValueError("Model reconstruction contains NaN or infinite values.")

    return reconstruction, embedding


# ============================================================
# Training and validation
# ============================================================

def run_training_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
):
    model.train()

    total_loss = 0.0
    total_samples = 0
    report_every = max(1, len(loader) // 10)
    epoch_start = datetime.now()

    for batch_index, batch in enumerate(loader, start=1):
        x, target, _ = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        reconstruction, _ = forward_model(model, x, target)
        loss = criterion(reconstruction, target)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered.")

        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_index % report_every == 0 or batch_index == len(loader):
            elapsed = (datetime.now() - epoch_start).total_seconds()
            print(
                f"Epoch [{epoch:03d}/{total_epochs}] "
                f"training batch [{batch_index}/{len(loader)}] "
                f"| elapsed: {elapsed:.1f}s",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("Training loader produced no samples.")

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, stage="Evaluation"):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    report_every = max(1, len(loader) // 5)
    stage_start = datetime.now()

    for batch_index, batch in enumerate(loader, start=1):
        x, target, _ = unpack_batch(batch, device)
        reconstruction, _ = forward_model(model, x, target)

        loss = criterion(reconstruction, target)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation/test loss encountered.")

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_index % report_every == 0 or batch_index == len(loader):
            elapsed = (datetime.now() - stage_start).total_seconds()
            print(
                f"{stage} batch [{batch_index}/{len(loader)}] "
                f"| elapsed: {elapsed:.1f}s",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError(f"{stage} loader produced no samples.")

    return total_loss / total_samples


# ============================================================
# Reconstruction metrics
# ============================================================

def calculate_metrics(target, prediction):
    """
    Calculate global metrics over all target elements.

    Inputs may have shape (N, ...). The target and prediction
    must be in the same representation and units.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    if target.shape != prediction.shape:
        raise ValueError(
            f"Metric shape mismatch: {target.shape} vs {prediction.shape}"
        )

    error = prediction - target

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    target_std = float(np.std(target))
    normalized_rmse = (
        rmse / target_std if target_std > 0 else None
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "normalized_rmse_by_target_std": normalized_rmse,
        "n_stars": int(target.shape[0]),
        "n_values": int(target.size),
    }


@torch.no_grad()
def collect_reconstructions(model, loader, device):
    """Collect per-star reconstructions, targets and identifiers."""
    model.eval()

    all_targets = []
    all_predictions = []
    all_inputs = []
    all_kics = []

    report_every = max(1, len(loader) // 5)
    stage_start = datetime.now()
    for batch_index, batch in enumerate(loader, start=1):
        x, target, kics = unpack_batch(batch, device)
        reconstruction, _ = forward_model(model, x, target)

        x_np = x.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        prediction_np = reconstruction.detach().cpu().numpy()

        if torch.is_tensor(kics):
            kic_values = kics.detach().cpu().numpy().reshape(-1).tolist()
        elif isinstance(kics, (list, tuple, np.ndarray)):
            kic_values = list(np.asarray(kics).reshape(-1))
        else:
            kic_values = [kics]

        all_inputs.extend(list(x_np))
        all_targets.extend(list(target_np))
        all_predictions.extend(list(prediction_np))
        all_kics.extend(kic_values)

        if batch_index % report_every == 0 or batch_index == len(loader):
            elapsed = (datetime.now() - stage_start).total_seconds()
            print(
                f"Collecting reconstructions "
                f"[{batch_index}/{len(loader)}] | elapsed: {elapsed:.1f}s",
                flush=True,
            )

    if not all_targets:
        raise ValueError("No reconstructions were collected.")

    return (
        np.asarray(all_inputs),
        np.asarray(all_targets),
        np.asarray(all_predictions),
        all_kics
    )


def per_star_metrics(kics, targets, predictions):
    rows = []

    for i, kic in enumerate(kics):
        target = np.asarray(targets[i], dtype=np.float64)
        prediction = np.asarray(predictions[i], dtype=np.float64)
        error = prediction - target

        rows.append({
            "kic": int(kic),
            "mse": float(np.mean(error ** 2)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
        })

    return pd.DataFrame(rows)


def calculate_constant_baseline(train_targets, test_targets):
    """
    Baseline: predict the global mean of all training target values
    for every output element. This is a simple reference, not a
    substitute for a domain-specific interpolation baseline.
    """
    mean_value = float(np.mean(train_targets))

    prediction = np.full_like(
        test_targets,
        fill_value=mean_value,
        dtype=np.float64
    )

    return calculate_metrics(test_targets, prediction)


# ============================================================
# Plots
# ============================================================

def plot_training_history(history_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training loss"
    )
    ax.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE reconstruction loss")
    ax.set_title("Autoencoder training history")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "training_history.png"),
        dpi=160
    )
    plt.close(fig)


def plot_reconstruction_examples(
    inputs,
    targets,
    predictions,
    kics,
    output_dir,
    num_examples
):
    os.makedirs(output_dir, exist_ok=True)

    n = min(num_examples, len(targets))

    for i in range(n):
        target = np.asarray(targets[i]).reshape(-1)
        prediction = np.asarray(predictions[i]).reshape(-1)
        error = target - prediction

        if len(target) != len(prediction):
            raise ValueError("Target and prediction lengths differ.")

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        axes[0].plot(target, label="Target", linewidth=1)
        axes[0].plot(
            prediction,
            label="Reconstruction",
            linewidth=1,
            alpha=0.85
        )
        axes[0].set_ylabel("Signal")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        axes[1].plot(error, linewidth=0.8)
        axes[1].axhline(0, linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Output index")
        axes[1].set_ylabel("Target − reconstruction")
        axes[1].grid(alpha=0.2)

        mse = float(np.mean(error ** 2))
        mae = float(np.mean(np.abs(error)))

        fig.suptitle(
            f"KIC {kics[i]} | Test MSE={mse:.5g} | MAE={mae:.5g}"
        )
        fig.tight_layout()

        fig.savefig(
            os.path.join(
                output_dir,
                f"kic_{int(kics[i])}_reconstruction.png"
            ),
            dpi=160
        )
        plt.close(fig)


def plot_error_distribution(per_star_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_star_df["mae"], bins=40)
    ax.set_xlabel("Per-star MAE")
    ax.set_ylabel("Number of stars")
    ax.set_title("Test reconstruction error distribution")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "test_mae_distribution.png"),
        dpi=160
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def train_autoencoder(config_path):
    print(f"Loading configuration: {config_path}", flush=True)
    config = load_config(config_path)
    settings = get_settings(config)

    set_seed(settings["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}", flush=True)
    print(f"Loaded config: {config['_config_path']}", flush=True)
    print(
        "Settings: "
        f"input_length={settings['input_length']}, "
        f"reconstruction_length={settings['reconstruction_length']}, "
        f"batch_size={settings['batch_size']}, "
        f"epochs={settings['epochs']}, "
        f"workers={settings['num_workers']}",
        flush=True,
    )

    # --------------------------------------------------------
    # Experiment directory
    # --------------------------------------------------------

    output_root = resolve_path(
        settings["output_dir"],
        config["_config_path"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"autoencoder_{timestamp}"
    experiment_dir = output_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {experiment_dir}", flush=True)

    reconstruction_dir = experiment_dir / "reconstruction_examples"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(settings["checkpoint_name"]).name
    checkpoint_path = experiment_dir / checkpoint_name

    # Save the exact config, without internal helper fields.
    saved_config = copy.deepcopy(config)
    saved_config.pop("_config_path", None)

    with (experiment_dir / "config.yaml").open(
        "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(saved_config, f, sort_keys=False)

    # --------------------------------------------------------
    # Data and splits
    # --------------------------------------------------------

    print("Loading and filtering manifest...", flush=True)
    df, manifest_path = prepare_manifest(config, settings)
    print("Splitting stars into train/validation/test sets...", flush=True)
    partitions, split_dfs = split_by_kic(df, settings)

    split_records = []
    for split_name, kics in partitions.items():
        split_records.extend({
            "kic": int(kic),
            "split": split_name
        } for kic in kics)

    pd.DataFrame(split_records).to_csv(
        experiment_dir / "split_kics.csv",
        index=False
    )

    train_dataset = make_dataset(
        split_dfs["train"],
        settings,
        deterministic=False
    )
    val_dataset = make_dataset(
        split_dfs["validation"],
        settings,
        deterministic=True
    )
    test_dataset = make_dataset(
        split_dfs["test"],
        settings,
        deterministic=True
    )

    loader_seed = settings["seed"]

    train_loader = make_loader(
        train_dataset,
        settings,
        shuffle=True,
        seed=loader_seed
    )
    val_loader = make_loader(
        val_dataset,
        settings,
        shuffle=False,
        seed=loader_seed + 1
    )
    test_loader = make_loader(
        test_dataset,
        settings,
        shuffle=False,
        seed=loader_seed + 2
    )

    print(
        "Data loaders ready: "
        f"train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"test={len(test_dataset)} samples; "
        f"batches per epoch={len(train_loader)}",
        flush=True,
    )
    print("Loading one sample to verify dataset and tensor shapes...", flush=True)
    sample_x, sample_target, sample_kic = train_dataset[0]

    if not torch.is_tensor(sample_x):
        sample_x = torch.as_tensor(sample_x)
    if not torch.is_tensor(sample_target):
        sample_target = torch.as_tensor(sample_target)

    print(f"Example input shape: {tuple(sample_x.shape)}", flush=True)
    print(f"Example target shape: {tuple(sample_target.shape)}", flush=True)
    print(f"Configured input length: {settings['input_length']}", flush=True)
    print(
        "Configured reconstruction length:",
        settings["reconstruction_length"],
        flush=True,
    )

    # --------------------------------------------------------
    # Model, loss, optimizer and scheduler
    # --------------------------------------------------------

    model = build_model(settings, device)
    print("Autoencoder model initialized.", flush=True)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"],
        min_lr=settings["scheduler_min_lr"]
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, settings["epochs"] + 1):
        train_loss = run_training_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            settings["epochs"],
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            stage=f"Epoch [{epoch:03d}/{settings['epochs']}] validation",
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = (
            val_loss < best_val_loss - settings["min_delta"]
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": saved_config,
                    "seed": settings["seed"],
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_type": "CNNBiLSTMAutoencoder",
                    "input_length": settings["input_length"],
                    "reconstruction_length": settings["reconstruction_length"],
                },
                checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        })

        print(
            f"Epoch [{epoch:03d}/{settings['epochs']}] "
            f"| Train: {train_loss:.6f} "
            f"| Val: {val_loss:.6f} "
            f"| LR: {current_lr:.2e}"
            + (" | BEST" if improved else ""),
            flush=True,
        )

        if epochs_without_improvement >= settings["patience"]:
            print(
                "Early stopping: validation loss did not improve "
                f"for {settings['patience']} epochs.",
                flush=True,
            )
            break

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Training ended without producing a best-model checkpoint."
        )

    print("Training complete; saving history and plot...", flush=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        experiment_dir / "training_history.csv",
        index=False
    )
    plot_training_history(history_df, str(experiment_dir))

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_loss = evaluate_loss(
        model,
        test_loader,
        criterion,
        device,
        stage="Test evaluation",
    )

    (
        _test_inputs,
        test_targets,
        test_predictions,
        test_kics
    ) = collect_reconstructions(
        model,
        test_loader,
        device
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_predictions
    )

    test_metrics["mse"] = float(test_loss)

    # Per-star test metrics.
    per_star_df = per_star_metrics(
        test_kics,
        test_targets,
        test_predictions
    )
    per_star_df.to_csv(
        experiment_dir / "per_star_test_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Simple constant baseline
    # --------------------------------------------------------

    print("Computing constant-mean baseline...", flush=True)
    train_targets = []
    report_every = max(1, len(train_loader) // 5)
    baseline_start = datetime.now()
    for batch_index, batch in enumerate(train_loader, start=1):
        _, target, _ = unpack_batch(batch, device)
        train_targets.append(target.detach().cpu().numpy())
        if batch_index % report_every == 0 or batch_index == len(train_loader):
            elapsed = (datetime.now() - baseline_start).total_seconds()
            print(
                f"Baseline target batch "
                f"[{batch_index}/{len(train_loader)}] "
                f"| elapsed: {elapsed:.1f}s",
                flush=True,
            )

    train_targets = np.concatenate(train_targets, axis=0)

    baseline_metrics = calculate_constant_baseline(
        train_targets,
        test_targets
    )

    # --------------------------------------------------------
    # Reconstruction plots
    # --------------------------------------------------------

    plot_reconstruction_examples(
        _test_inputs,
        test_targets,
        test_predictions,
        test_kics,
        str(reconstruction_dir),
        settings["num_examples"]
    )

    plot_error_distribution(
        per_star_df,
        str(experiment_dir)
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "experiment_name": experiment_name,
        "model_type": "CNNBiLSTMAutoencoder",
        "manifest_path": manifest_path,
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val_loss),
        "test_reconstruction": test_metrics,
        "constant_mean_baseline": baseline_metrics,
        "per_star_test_summary": {
            "mean_mse": float(per_star_df["mse"].mean()),
            "median_mse": float(per_star_df["mse"].median()),
            "mean_rmse": float(per_star_df["rmse"].mean()),
            "median_rmse": float(per_star_df["rmse"].median()),
            "mean_mae": float(per_star_df["mae"].mean()),
            "median_mae": float(per_star_df["mae"].median()),
        },
    }

    with (experiment_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    with (experiment_dir / "metrics.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("AUTOENCODER EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: CNNBiLSTMAutoencoder\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation MSE: {best_val_loss:.8f}\n")
        f.write(f"Test MSE: {test_metrics['mse']:.8f}\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.8f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.8f}\n")
        f.write(
            "Test normalized RMSE: "
            f"{test_metrics['normalized_rmse_by_target_std']}\n\n"
        )

        f.write("CONSTANT-MEAN BASELINE\n")
        f.write("-" * 60 + "\n")
        f.write(f"MSE: {baseline_metrics['mse']:.8f}\n")
        f.write(f"RMSE: {baseline_metrics['rmse']:.8f}\n")
        f.write(f"MAE: {baseline_metrics['mae']:.8f}\n\n")

        f.write("DATA SPLIT\n")
        f.write("-" * 60 + "\n")
        for name, kics in partitions.items():
            f.write(f"{name}: {len(kics)} KICs\n")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config["_config_path"],
        "manifest_path": manifest_path,
        "output_directory": str(experiment_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": str(device),
        "seed": settings["seed"],
        "model_type": "CNNBiLSTMAutoencoder",
        "input_length": settings["input_length"],
        "reconstruction_length": settings["reconstruction_length"],
        "n_train": len(split_dfs["train"]),
        "n_validation": len(split_dfs["validation"]),
        "n_test": len(split_dfs["test"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "epochs_completed": len(history),
        "early_stopping_patience": settings["patience"],
        "early_stopping_min_delta": settings["min_delta"],
    }

    with (experiment_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70, flush=True)
    print("AUTOENCODER TRAINING SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation MSE: {best_val_loss:.6f}", flush=True)
    print(f"Test MSE: {test_metrics['mse']:.6f}", flush=True)
    print(f"Test RMSE: {test_metrics['rmse']:.6f}", flush=True)
    print(f"Test MAE: {test_metrics['mae']:.6f}", flush=True)
    print(f"Baseline MSE: {baseline_metrics['mse']:.6f}", flush=True)
    print(f"Results saved to: {experiment_dir.resolve()}", flush=True)
    print("Autoencoder training complete.", flush=True)

    return str(experiment_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the CNN-BiLSTM autoencoder."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASELINE_DIR / "config.yaml"),
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    train_autoencoder(args.config)
