import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models import CNNBiLSTMAutoencoder
from utils.reconstruction_dataset import AstroReconstructionDataset


# ============================================================
# Configuration
# ============================================================

with open(
    "./Baseline/config.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# Reproducibility
# ============================================================

seed = config["training"]["seed"]

np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

manifest_path = config["data"]["manifest_path"]

df = pd.read_csv(
    manifest_path
)

# Keep the same duration filtering as the
# supervised experiment.

if config["data"]["filter_duration"] is not None:

    df = df[
        df["duration"]
        == config["data"]["filter_duration"]
    ].copy()


# Remove duplicate KICs if necessary.

df = df.drop_duplicates(
    subset=["kic"]
).reset_index(drop=True)


print(
    "Number of stars:",
    len(df)
)


# ============================================================
# KIC-level split
# ============================================================

kics = df["kic"].unique()

train_kics, test_kics = train_test_split(
    kics,
    test_size=0.20,
    random_state=seed
)

train_kics, val_kics = train_test_split(
    train_kics,
    test_size=0.125,
    random_state=seed
)

# Gives approximately:
#
# 70% train
# 10% validation
# 20% test


train_df = df[
    df["kic"].isin(train_kics)
].reset_index(drop=True)

val_df = df[
    df["kic"].isin(val_kics)
].reset_index(drop=True)

test_df = df[
    df["kic"].isin(test_kics)
].reset_index(drop=True)


print(
    f"Train: {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Test: {len(test_df)}"
)


# ============================================================
# Datasets
# ============================================================

reconstruction_length = config[
    "autoencoder"
]["reconstruction_length"]


train_dataset = AstroReconstructionDataset(
    train_df,
    target_length=65000,
    reconstruction_length=reconstruction_length,
    deterministic=False
)

val_dataset = AstroReconstructionDataset(
    val_df,
    target_length=65000,
    reconstruction_length=reconstruction_length,
    deterministic=True
)

test_dataset = AstroReconstructionDataset(
    test_df,
    target_length=65000,
    reconstruction_length=reconstruction_length,
    deterministic=True
)


# ============================================================
# DataLoaders
# ============================================================

batch_size = config[
    "autoencoder"
]["batch_size"]

num_workers = config[
    "autoencoder"
]["num_workers"]


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Model
# ============================================================

model = CNNBiLSTMAutoencoder(
    hidden_size=config[
        "autoencoder"
    ]["hidden_size"],

    num_layers=config[
        "autoencoder"
    ]["num_layers"],

    reconstruction_length=reconstruction_length
)

model = model.to(device)


# ============================================================
# Loss + optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["autoencoder"]["learning_rate"]
)


# ============================================================
# Training
# ============================================================

epochs = config[
    "autoencoder"
]["epochs"]

patience = config[
    "autoencoder"
]["early_stopping"]["patience"]

min_delta = config[
    "autoencoder"
]["early_stopping"]["min_delta"]


best_val_loss = float("inf")
epochs_without_improvement = 0


os.makedirs(
    "./Baseline/results/autoencoder",
    exist_ok=True
)

checkpoint_path = os.path.join(
    "./Baseline/results/autoencoder",
    config["autoencoder"]["checkpoint_name"]
)


for epoch in range(epochs):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for x, target, _ in train_loader:

        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        reconstruction, embedding = model(x)

        loss = criterion(
            reconstruction,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * x.size(0)
        )

    train_loss /= len(train_dataset)


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, target, _ in val_loader:

            x = x.to(device)
            target = target.to(device)

            reconstruction, embedding = model(x)

            loss = criterion(
                reconstruction,
                target
            )

            val_loss += (
                loss.item()
                * x.size(0)
            )

    val_loss /= len(val_dataset)


    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train loss: {train_loss:.6f} | "
        f"Val loss: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss - min_delta:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "config":
                    config,

                "best_val_loss":
                    best_val_loss
            },
            checkpoint_path
        )

        print(
            "  Saved new best model."
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            "Early stopping."
        )

        break


# ============================================================
# Load best model
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Test
# ============================================================

model.eval()

test_loss = 0.0

with torch.no_grad():

    for x, target, _ in test_loader:

        x = x.to(device)
        target = target.to(device)

        reconstruction, embedding = model(x)

        loss = criterion(
            reconstruction,
            target
        )

        test_loss += (
            loss.item()
            * x.size(0)
        )

test_loss /= len(test_dataset)


print(
    f"Test reconstruction loss: "
    f"{test_loss:.6f}"
)