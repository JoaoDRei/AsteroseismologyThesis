#!/usr/bin/env python3
"""Compare frozen autoencoder embeddings, PCA, and pooled light curves.

Place in Baseline/analysis and run from the thesis root. By default, compares
the matched seed-42 F9 and F12 runs. Each --run argument overrides those
defaults and uses LABEL=RUN_DIR format. Runs require
latent_analysis/latent_vectors.csv; generate it with analyze_autoencoder.py.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from Baseline.analysis.analyze_autoencoder import (
    load_manifest,
    load_splits,
    read_yaml,
    resolve_manifest,
    split_signature,
)
from Baseline.paths import BASELINE_DIR
from Baseline.utils.reconstruction_dataset import AstroReconstructionDataset


SPLITS = ("train", "validation", "test")
GRID = (
    0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
    10000.0, 100000.0, 1000000.0,
)
DEFAULT_RUNS = (
    "F9=Baseline/results/autoencoder_20261110_150132_001",
    "F12=Baseline/results/autoencoder_20261110_154956",
)


def load_run(run_dir):
    config = read_yaml(run_dir / "config.yaml")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    ae = config["autoencoder"]
    protocol = (
        config["data"].get("filter_duration"),
        int(metadata.get("input_length", ae.get(
            "input_length", config["data"].get("target_length", 65000)
        ))),
        int(ae["reconstruction_length"]),
    )
    if not ae.get("pooled_input", False) or protocol != ("80d", 65000, 4096):
        raise ValueError(f"{run_dir}: expected 80d, 65000-point crop, pooled 4096-point input.")
    return config, metadata, load_splits(run_dir / "split_kics.csv").set_index("kic").sort_index()


def load_embedding(run_dir, splits):
    path = run_dir / "latent_analysis" / "latent_vectors.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run python -m Baseline.analysis.analyze_autoencoder "
            f"--run-dir {run_dir} first."
        )
    frame = pd.read_csv(path).set_index("kic", verify_integrity=True).sort_index()
    if not frame["split"].equals(splits["split"]):
        raise ValueError(f"{path}: KICs or split assignments differ from the saved split.")
    columns = sorted(c for c in frame if re.fullmatch(r"latent_\d+", c))
    values = frame[columns].to_numpy(dtype=np.float32)
    if len(columns) != 64 or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected 64 finite latent features per star.")
    return values


def pooled_curves(frame):
    dataset = AstroReconstructionDataset(
        frame, target_length=65000, reconstruction_length=4096,
        deterministic=True, pooled_input=True,
    )
    curves = np.empty((len(frame), 4096), dtype=np.float32)
    offset = 0
    for _, target, kics in DataLoader(dataset, batch_size=64, shuffle=False):
        end = offset + len(kics)
        if not np.array_equal(kics.numpy(), frame["kic"].iloc[offset:end].to_numpy()):
            raise ValueError("Dataset KIC order differs from the saved split.")
        curves[offset:end] = target.numpy()
        offset = end
    if offset != len(frame) or not np.isfinite(curves).all():
        raise ValueError("Expected one finite pooled curve per KIC.")
    return curves


def labels_for(frame, target, override):
    if target == "class":
        candidates = ("class", "evolstate", "EvolState")
    else:
        candidates = (f"truth_log_{target}", target)
    column = override or next((name for name in candidates if name in frame), None)
    if column is None or column not in frame:
        raise ValueError(f"Missing label column for {target}. Use --{target.replace('_', '-')}-column.")
    series = frame[column]
    if target == "class":
        text = series.astype("string").str.strip().str.upper()
        numeric = pd.to_numeric(series, errors="coerce").where(lambda s: s.isin([0, 1]))
        labels = text.map({"RGB": 0, "RC": 1}).fillna(numeric)
    else:
        labels = pd.to_numeric(series, errors="coerce")
        if not column.startswith("truth_log_"):
            labels = np.log10(labels.where(labels > 0))
    return labels.to_numpy(dtype=float, na_value=np.nan), column


def fit_probe(features, labels, masks, classification):
    train, validation, test = masks
    scaler = StandardScaler().fit(features[train])
    x_train, x_val, x_test = (scaler.transform(features[m]) for m in masks)
    y_train, y_val, y_test = (labels[m] for m in masks)
    best = None
    for penalty in GRID:
        if classification:
            model = LogisticRegression(
                C=1.0 / penalty, class_weight="balanced",
                solver="lbfgs", max_iter=4000,
            )
        else:
            model = Ridge(alpha=penalty, solver="lsqr", tol=1e-6)
        model.fit(x_train, y_train)
        val_prediction = model.predict(x_val)
        score = (balanced_accuracy_score(y_val, val_prediction) if classification
                 else mean_absolute_error(y_val, val_prediction))
        loss = 1.0 - score if classification else score
        if best is None or loss < best[0]:
            best = loss, penalty, model, score
    _, penalty, model, val_score = best
    predictions = model.predict(x_test)
    test_score = (balanced_accuracy_score(y_test, predictions) if classification
                  else mean_absolute_error(y_test, predictions))
    probability = model.predict_proba(x_test)[:, 1] if classification else None
    return penalty, val_score, test_score, predictions, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=RUN_DIR",
        help=(
            "Run to compare; may be supplied more than once. "
            "Defaults to the matched seed-42 F9 and F12 runs."
        ),
    )
    parser.add_argument("--manifest", help="Override the manifest path recorded by the runs.")
    parser.add_argument("--nu-max-column", help="Column in microhertz, or truth_log_* in dex.")
    parser.add_argument("--delta-nu-column", help="Column in microhertz, or truth_log_* in dex.")
    parser.add_argument("--class-column", help="Column with RGB/RC or 0/1 (RGB=0, RC=1).")
    parser.add_argument("--output-dir", type=Path,
                        default=BASELINE_DIR / "results_analysis" / "frozen_probes")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for linear algebra.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    features = {}
    runs = {}
    splits = None
    manifest_path = None
    for item in args.run or DEFAULT_RUNS:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory:
            parser.error("--run must be LABEL=RUN_DIR")
        if name in features or name in {"pca64", "pooled4096"}:
            parser.error(f"Duplicate or reserved representation label: {name}")
        run_dir = Path(directory).expanduser().resolve()
        config, metadata, run_split = load_run(run_dir)
        path = resolve_manifest(config, run_dir, metadata, args.manifest)
        if splits is None:
            splits, manifest_path = run_split, path
            manifest = load_manifest(path, config)
        if not splits["split"].equals(run_split["split"]) or path != manifest_path:
            raise ValueError(f"{run_dir}: manifest or saved KIC split differs from the first run.")
        features[name] = load_embedding(run_dir, splits)
        runs[name] = str(run_dir)

    groups = splits["split"].to_numpy()
    if set(groups) != set(SPLITS):
        raise ValueError("Expected nonempty train, validation, and test splits.")
    frame = manifest.set_index("kic").loc[splits.index].reset_index()
    tasks = {}
    for target in ("nu_max", "delta_nu", "class"):
        labels, column = labels_for(frame, target, getattr(args, f"{target}_column"))
        masks = tuple((groups == split) & np.isfinite(labels) for split in SPLITS)
        if any(not mask.any() for mask in masks):
            raise ValueError(f"{target} ({column}): at least one split has no usable labels.")
        if target == "class" and any(len(np.unique(labels[m])) != 2 for m in masks):
            raise ValueError(f"{column}: each split needs both RGB and RC stars.")
        tasks[target] = labels, column, masks

    print("Loading deterministic pooled light curves.", flush=True)
    curves = pooled_curves(frame)
    train = groups == "train"
    if train.sum() <= 64:
        raise ValueError("PCA requires at least 65 training stars for 64 components.")
    rows, per_star = [], []
    with threadpool_limits(limits=args.threads):
        print("Fitting 64-component PCA on training curves.", flush=True)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=42)
        pca.fit(curves[train])
        features["pca64"] = pca.transform(curves)
        features["pooled4096"] = curves

        for target, (labels, column, masks) in tasks.items():
            classification = target == "class"
            if not classification:
                train, validation, test = masks
                median = float(np.median(labels[train]))
                rows.append({
                    "representation": "train_median",
                    "target": target,
                    "label_column": column,
                    "n_features": 0,
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "n_test": int(test.sum()),
                    "metric": "mae_dex",
                    "validation_score": float(np.mean(np.abs(labels[validation] - median))),
                    "test_score": float(np.mean(np.abs(labels[test] - median))),
                })
            for name, values in features.items():
                print(f"Fitting {target} probe on {name}.", flush=True)
                penalty, val_score, test_score, pred, probability = fit_probe(
                    values, labels, masks, classification
                )
                rows.append({
                    "representation": name, "target": target, "label_column": column,
                    "n_features": values.shape[1],
                    "n_train": int(masks[0].sum()), "n_validation": int(masks[1].sum()),
                    "n_test": int(masks[2].sum()),
                    "metric": "balanced_accuracy" if classification else "mae_dex",
                    "parameter": "C" if classification else "alpha",
                    "parameter_value": 1.0 / penalty if classification else penalty,
                    "validation_score": val_score, "test_score": test_score,
                })
                test = masks[2]
                per_star.append(pd.DataFrame({
                    "kic": frame.loc[test, "kic"].to_numpy(), "target": target,
                    "representation": name, "truth": labels[test],
                    "prediction": pred, "probability_rc": probability,
                }))

    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "probe_summary.csv", index=False)
    pd.concat(per_star, ignore_index=True).to_csv(output / "probe_test_predictions.csv", index=False)
    (output / "probe_metadata.json").write_text(json.dumps({
        "manifest": str(manifest_path), "runs": runs,
        "split_signature": split_signature(splits.index, groups),
        "ridge_alpha_grid": GRID, "logistic_c_grid": [1.0 / a for a in GRID],
        "pca_components": 64, "pca_random_state": 42,
        "pca_training_stars": int(train.sum()),
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
