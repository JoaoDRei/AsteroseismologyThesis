# cross_validate.py
#
# Standalone K-fold cross-validation script.
# Does NOT modify or import the training procedure from train.py.
#
# It uses the same:
#   - manifest
#   - AstroBaselineDataset
#   - models
#   - losses
#   - optimizer
#   - scheduler
#   - early stopping
# as train.py.
#
# The important difference is that the data are split by KIC into
# K folds, so different light-curve rows belonging to the same star
# can never appear in different folds.

import os
import copy
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from utils import AstroBaselineDataset, SyntheticPSDPeakDataset
from models import SimpleMLP, SimpleCNN, SimpleTransformer, ResNet1D, MobileNet1D, CNNLSTM, AstroConformer, AstroConformerV2


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Loss functions
# ============================================================

def get_loss_fn(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "huber":
        return nn.SmoothL1Loss()
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


# ============================================================
# Model construction
# ============================================================

def build_model(config, dataset, targets_list):
    model_type = config["model"]["type"]

    if model_type == "cnn":
        model = SimpleCNN(
            output_dim=len(targets_list)
        )

    elif model_type == "mlp":
        model = SimpleMLP(
            input_size=dataset.target_length,
            output_dim=len(targets_list)
        )

    elif model_type == "transformer":
        model = SimpleTransformer(
            seq_length=config["data"]["seq_length"],
            output_dim=len(targets_list)
        )
    elif config['model']['type'] == "resnet1d":
        model = ResNet1D(output_dim=len(targets_list))

    elif config['model']['type'] == "mobilenet1d":
        model = MobileNet1D(output_dim=len(targets_list))


    elif config['model']['type'] == "cnnlstm":
        model = CNNLSTM(output_dim=len(targets_list))

    elif config['model']['type'] == "astroconformer":
        model = AstroConformer(
            encoder_dim=config['model']['encoder_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            patch_size=config['model']['patch_size'],
            conv_kernel_size=config['model']['conv_kernel_size'],
            dropout=config['model']['dropout'],
            output_dim=len(targets_list)
        )

    elif config['model']['type'] == "astroconformer_v2":
        model = AstroConformerV2(
            encoder_dim=config['model']['encoder_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            patch_size=config['model']['patch_size'],
            conv_kernel_size=config['model']['conv_kernel_size'],
            dropout=config['model']['dropout'],
            output_dim=len(targets_list)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# ============================================================
# One epoch of training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fns,
    loss_weights,
    targets_list,
    device,
):
    model.train()

    total_loss = 0.0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        preds = model(x)

        loss = 0.0

        for idx, target in enumerate(targets_list):
            pred = preds[:, idx]
            truth = y[:, idx]

            loss_t = loss_fns[target](pred, truth)
            loss += loss_weights[target] * loss_t

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if len(loader) == 0:
        raise ValueError("Training loader is empty.")

    return total_loss / len(loader)


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    loss_fns,
    loss_weights,
    targets_list,
    device,
):
    model.eval()

    total_loss = 0.0

    all_preds = {
        target: []
        for target in targets_list
    }

    all_truth = {
        target: []
        for target in targets_list
    }

    with torch.no_grad():

        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            loss = 0.0

            for idx, target in enumerate(targets_list):

                pred = preds[:, idx]
                truth = y[:, idx]

                loss_t = loss_fns[target](pred, truth)
                loss += loss_weights[target] * loss_t

                all_preds[target].extend(
                    pred.cpu().numpy()
                )

                all_truth[target].extend(
                    truth.cpu().numpy()
                )

            total_loss += loss.item()

    if len(loader) == 0:
        raise ValueError("Validation loader is empty.")

    avg_loss = total_loss / len(loader)

    return avg_loss, all_preds, all_truth


# ============================================================
# Classification metrics
# ============================================================

def classification_metrics(preds, truth):
    logits = np.asarray(preds)
    truth = np.asarray(truth).astype(int)

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predicted_classes = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(
        truth,
        predicted_classes
    )

    balanced_accuracy = balanced_accuracy_score(
        truth,
        predicted_classes
    )

    f1 = f1_score(
        truth,
        predicted_classes,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


# ============================================================
# Main cross-validation procedure
# ============================================================

def cross_validate(config_path="./Baseline/config.yaml"):

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")

    # --------------------------------------------------------
    # 2. Targets
    # --------------------------------------------------------

    synthetic = config.get("debug", {}).get(
        "synthetic",
        False
    )

    if synthetic:
        targets_list = ["peak_loc"]
    else:
        targets_list = config["model"]["targets"]

    print(f"Targets: {targets_list}")

    # --------------------------------------------------------
    # 3. Loss functions
    # --------------------------------------------------------

    loss_fns = {
        target: get_loss_fn(
            config["model"]["loss_functions"][target]
        )
        for target in targets_list
    }

    loss_weights = config["model"]["loss_weights"]

    # --------------------------------------------------------
    # 4. Load data
    # --------------------------------------------------------

    if synthetic:

        print("\nUsing synthetic dataset.")

        full_ds = SyntheticPSDPeakDataset(
            n_samples=2000,
            length=config["data"]["target_length"]
        )

        all_indices = np.arange(len(full_ds))

        # Synthetic data has no stellar class, so ordinary KFold.
        splitter = KFold(
            n_splits=config["training"]["n_folds"],
            shuffle=True,
            random_state=seed
        )

        split_labels = None

    else:

        manifest_path = config["data"]["manifest_path"]

        print(
            f"\nLoading manifest from "
            f"{manifest_path}..."
        )

        df = pd.read_csv(manifest_path)

        # ----------------------------------------------------
        # Normalize RGB / RC exactly as train.py does.
        #
        # This is NOT replacing dataset.py's target handling.
        # We need evolstate_norm here only because the splitter
        # needs the labels before constructing the Dataset.
        # ----------------------------------------------------

        if "class" in targets_list:

            def normalize_ev(value):
                if pd.isna(value):
                    return None

                value = str(value).strip().upper()

                if value in ["RGB", "RC"]:
                    return value

                return None

            df["evolstate_norm"] = (
                df["evolstate"]
                .apply(normalize_ev)
            )

            before = len(df)

            df = df[
                df["evolstate_norm"].notnull()
            ].reset_index(drop=True)

            after = len(df)

            print(
                f"Removed {before - after} rows "
                f"with invalid/missing evolutionary state."
            )

        # ----------------------------------------------------
        # Duration filtering
        # ----------------------------------------------------

        filter_duration = config["data"].get(
            "filter_duration"
        )

        if filter_duration:

            print(
                f"Filtering duration: "
                f"{filter_duration}"
            )

            df = df[
                df["duration"] == filter_duration
            ].reset_index(drop=True)

        # ----------------------------------------------------
        # Optional sample size
        #
        # IMPORTANT:
        # sample_size refers to unique KICs, just like train.py.
        # ----------------------------------------------------

        sample_size = config["data"].get(
            "sample_size"
        )

        unique_kics = df["kic"].unique()

        rng = np.random.default_rng(seed)
        rng.shuffle(unique_kics)

        if sample_size is not None:

            unique_kics = unique_kics[
                :sample_size
            ]

            df = df[
                df["kic"].isin(unique_kics)
            ].reset_index(drop=True)

        print(
            f"Unique KICs available: "
            f"{df['kic'].nunique()}"
        )

        # ----------------------------------------------------
        # Create Dataset AFTER all filtering.
        # ----------------------------------------------------

        full_ds = AstroBaselineDataset(
            df,
            mode=config["data"]["mode"],
            target_length=config["data"]["target_length"]
        )

        # ----------------------------------------------------
        # VERY IMPORTANT:
        #
        # The manifest has multiple rows per KIC
        # (20d / 55d / 80d).
        #
        # We therefore split UNIQUE KICs, not rows.
        # ----------------------------------------------------

        unique_kics = df["kic"].unique()

        kic_to_label = None

        if "class" in targets_list:

            # Each KIC should have one evolutionary state.
            kic_labels_df = (
                df[
                    ["kic", "evolstate_norm"]
                ]
                .drop_duplicates(
                    subset=["kic"]
                )
            )

            kic_to_label = dict(
                zip(
                    kic_labels_df["kic"],
                    kic_labels_df["evolstate_norm"]
                )
            )

            split_labels = np.array([
                0 if kic_to_label[kic] == "RGB"
                else 1
                for kic in unique_kics
            ])

            splitter = StratifiedKFold(
                n_splits=config["training"]["n_folds"],
                shuffle=True,
                random_state=seed
            )

        else:

            split_labels = None

            splitter = KFold(
                n_splits=config["training"]["n_folds"],
                shuffle=True,
                random_state=seed
            )

        all_indices = np.arange(
            len(unique_kics)
        )

    # ========================================================
    # 5. Cross-validation
    # ========================================================

    n_folds = config["training"]["n_folds"]

    print(
        f"\nNumber of folds: {n_folds}"
    )

    fold_results = []

    # Store predictions/metrics if classification is used.
    classification_results = []

    for fold, (train_kic_idx, val_kic_idx) in enumerate(
        splitter.split(
            all_indices,
            split_labels
        ),
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}/{n_folds}")
        print("=" * 70)

        # ----------------------------------------------------
        # Get actual KIC IDs
        # ----------------------------------------------------

        if synthetic:

            train_idx = train_kic_idx.tolist()
            val_idx = val_kic_idx.tolist()

        else:

            train_kics = unique_kics[
                train_kic_idx
            ]

            val_kics = unique_kics[
                val_kic_idx
            ]

            # Convert KIC split back to row indices.
            #
            # This means every row belonging to a KIC stays
            # inside the same fold.
            train_idx = df[
                df["kic"].isin(train_kics)
            ].index.tolist()

            val_idx = df[
                df["kic"].isin(val_kics)
            ].index.tolist()

        print(
            f"Train rows: {len(train_idx)}"
        )
        print(
            f"Validation rows: {len(val_idx)}"
        )

        if not synthetic:
            print(
                f"Train KICs: {len(train_kics)}"
            )
            print(
                f"Validation KICs: {len(val_kics)}"
            )

            if "class" in targets_list:

                train_classes = df[
                    df["kic"].isin(train_kics)
                ]["evolstate_norm"].value_counts(
                    normalize=True
                )

                val_classes = df[
                    df["kic"].isin(val_kics)
                ]["evolstate_norm"].value_counts(
                    normalize=True
                )

                print("\nTrain class distribution:")
                print(train_classes)

                print("\nValidation class distribution:")
                print(val_classes)

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        num_workers = config["data"].get(
            "num_workers",
            0
        )

        train_loader = DataLoader(
            Subset(
                full_ds,
                train_idx
            ),
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            Subset(
                full_ds,
                val_idx
            ),
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=num_workers
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print(
            f"\nInitializing "
            f"{config['model']['type'].upper()}..."
        )

        model = build_model(
            config,
            full_ds,
            targets_list
        )

        model.to(device)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["model"]["learning_rate"]
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        es_cfg = config["training"].get(
            "early_stopping",
            {}
        )

        es_patience = es_cfg.get(
            "patience",
            15
        )

        es_min_delta = float(es_cfg.get(
            "min_delta",
            1e-4
        ))

        es_restore_best = es_cfg.get(
            "restore_best_weights",
            True
        )

        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        # ----------------------------------------------------
        # Number of epochs
        #
        # Supports both:
        #
        # training:
        #   epochs: 50
        #
        # and your old config:
        #
        # model:
        #   epochs: 50
        # ----------------------------------------------------

        epochs = config["training"].get(
            "epochs",
            config["model"].get("epochs", 50)
        )

        # ====================================================
        # Training loop
        # ====================================================

        for epoch in range(epochs):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fns=loss_fns,
                loss_weights=loss_weights,
                targets_list=targets_list,
                device=device
            )

            val_loss, val_preds, val_truth = validate(
                model=model,
                loader=val_loader,
                loss_fns=loss_fns,
                loss_weights=loss_weights,
                targets_list=targets_list,
                device=device
            )

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Train: {train_loss:.4f} "
                f"| Val: {val_loss:.4f} "
                f"| LR: {current_lr:.2e}"
            )

            # ------------------------------------------------
            # Save best model in memory
            # ------------------------------------------------

            if val_loss < best_val_loss - es_min_delta:

                best_val_loss = val_loss

                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                epochs_no_improve = 0

                print(
                    f"  ⭐ New best validation loss: "
                    f"{best_val_loss:.4f}"
                )

            else:

                epochs_no_improve += 1

                print(
                    f"  No improvement: "
                    f"{epochs_no_improve}/{es_patience}"
                )

                if epochs_no_improve >= es_patience:

                    print(
                        f"  ⏱️ Early stopping "
                        f"(patience={es_patience})"
                    )

                    break

        # ====================================================
        # Restore best model
        # ====================================================

        if (
            es_restore_best
            and best_model_state is not None
        ):
            model.load_state_dict(
                best_model_state
            )

        # ====================================================
        # Final validation of best model
        # ====================================================

        final_val_loss, final_preds, final_truth = validate(
            model=model,
            loader=val_loader,
            loss_fns=loss_fns,
            loss_weights=loss_weights,
            targets_list=targets_list,
            device=device
        )

        print(
            f"\nFold {fold} best validation loss: "
            f"{final_val_loss:.4f}"
        )

        fold_result = {
            "fold": fold,
            "val_loss": final_val_loss
        }

        # ----------------------------------------------------
        # Classification metrics
        # ----------------------------------------------------

        if "class" in targets_list:

            metrics = classification_metrics(
                final_preds["class"],
                final_truth["class"]
            )

            fold_result.update(metrics)

            classification_results.append(
                metrics
            )

            print("\nClassification metrics:")

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"F1: "
                f"{metrics['f1']:.4f}"
            )

            # Full report for this fold
            logits = np.asarray(
                final_preds["class"]
            )

            probabilities = (
                1.0 /
                (1.0 + np.exp(-logits))
            )

            predicted_classes = (
                probabilities >= 0.5
            ).astype(int)

            print("\nClassification report:")
            print(
                classification_report(
                    np.asarray(
                        final_truth["class"]
                    ).astype(int),
                    predicted_classes,
                    target_names=[
                        "RGB",
                        "RC"
                    ],
                    zero_division=0
                )
            )

        # ----------------------------------------------------
        # Regression metrics
        # ----------------------------------------------------

        if "nu_max" in targets_list:

            pred = np.asarray(
                final_preds["nu_max"]
            )

            truth = np.asarray(
                final_truth["nu_max"]
            )

            mse = np.mean(
                (pred - truth) ** 2
            )

            rmse = np.sqrt(mse)

            fold_result["nu_max_mse"] = mse
            fold_result["nu_max_rmse"] = rmse

            print(
                f"nu_max RMSE: {rmse:.4f}"
            )

        if "delta_nu" in targets_list:

            pred = np.asarray(
                final_preds["delta_nu"]
            )

            truth = np.asarray(
                final_truth["delta_nu"]
            )

            mse = np.mean(
                (pred - truth) ** 2
            )

            rmse = np.sqrt(mse)

            fold_result["delta_nu_mse"] = mse
            fold_result["delta_nu_rmse"] = rmse

            print(
                f"delta_nu RMSE: {rmse:.4f}"
            )

        fold_results.append(
            fold_result
        )

    # ========================================================
    # 6. Summary
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    print("\n")
    print("=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print("\nPer-fold results:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nMean ± standard deviation:")

    metric_columns = [
        column
        for column in results_df.columns
        if column != "fold"
    ]

    summary = {}

    for metric in metric_columns:

        mean = results_df[metric].mean()
        std = results_df[metric].std(
            ddof=1
        )

        summary[metric] = {
            "mean": mean,
            "std": std
        }

        print(
            f"  {metric}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    # ========================================================
    # 7. Save results
    # ========================================================

    output_dir = os.path.join(
        os.path.dirname(
            config["paths"]["checkpoint_dir"]
        ),
        "cross_validation"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results_df.to_csv(
        os.path.join(
            output_dir,
            "fold_results.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            output_dir,
            "summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "K-FOLD CROSS-VALIDATION SUMMARY\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Model: "
            f"{config['model']['type']}\n"
        )

        f.write(
            f"Mode: "
            f"{config['data']['mode']}\n"
        )

        f.write(
            f"Targets: "
            f"{targets_list}\n"
        )

        f.write(
            f"Folds: "
            f"{n_folds}\n"
        )

        f.write(
            f"Seed: "
            f"{seed}\n\n"
        )

        f.write(
            "PER-FOLD RESULTS\n"
        )
        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            results_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nMEAN ± STANDARD DEVIATION\n"
        )
        f.write(
            "-" * 70 + "\n"
        )

        for metric, values in summary.items():

            f.write(
                f"{metric}: "
                f"{values['mean']:.6f} "
                f"+/- "
                f"{values['std']:.6f}\n"
            )

    # Save the exact config used for CV.
    with open(
        os.path.join(
            output_dir,
            "config.yaml"
        ),
        "w"
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False
        )

    print(
        f"\nResults saved to: "
        f"{output_dir}"
    )

    print(
        "\nCross-validation complete."
    )


# ============================================================
# Command line entry point
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="K-fold cross-validation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./Baseline/config.yaml",
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    cross_validate(
        config_path=args.config
    )