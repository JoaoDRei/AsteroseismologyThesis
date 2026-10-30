"""Extract latent vectors and analyze reconstructions from a trained CNN-BiLSTM autoencoder.

Run from the thesis project root:
    python -m Baseline.analysis.analyze_autoencoder
    python -m Baseline.analysis.analyze_autoencoder --run-dir Baseline/results/autoencoder_YYYYMMDD_HHMMSS

Uses the exact KIC split saved by train_autoencoder.py. Writes latent vectors,
per-star reconstruction errors, PCA coordinates/plots, and test reconstructions.
By default, analyzes the latest completed autoencoder run and ignores
incomplete runs. A completed run has training metadata, evaluation metrics,
configuration, split, and a model checkpoint.
Also updates Baseline/results_analysis/autoencoder_runs.csv,
autoencoder_split_metrics.csv, and plots/autoencoder_test_vs_baseline.png.
UMAP is optional: add --umap if umap-learn is installed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from Baseline.models import CNNBiLSTMAutoencoder
from Baseline.paths import BASELINE_DIR, PROJECT_ROOT
from Baseline.utils.reconstruction_dataset import AstroReconstructionDataset


DEFAULT_RESULTS_DIR = BASELINE_DIR / "results"
RESULTS_ANALYSIS_DIR = BASELINE_DIR / "results_analysis"
RESULTS_ANALYSIS_PLOTS_DIR = RESULTS_ANALYSIS_DIR / "plots"


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return cfg


def resolve_manifest(config, run_dir, metadata, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = []
    if metadata.get("manifest_path"):
        candidates.append(Path(metadata["manifest_path"]))
    if config.get("data", {}).get("manifest_path"):
        candidates.append(Path(config["data"]["manifest_path"]))
    roots = [PROJECT_ROOT, Path.cwd(), run_dir, run_dir.parent]
    for candidate in candidates:
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        for root in roots:
            path = (root / candidate).resolve()
            if path.is_file():
                return path
    raise FileNotFoundError(
        "Cannot find manifest. Run from the project root or pass --manifest."
    )


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint  # raw state_dict
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def load_manifest(path, config):
    df = pd.read_csv(path)
    if "kic" not in df:
        raise ValueError("Manifest must contain a kic column.")
    kic = pd.to_numeric(df["kic"], errors="coerce")
    if kic.isna().any():
        raise ValueError("Manifest contains missing/non-numeric KIC values.")
    df["kic"] = kic.astype(np.int64)
    duration = config.get("data", {}).get("filter_duration")
    if duration is not None and "duration" in df:
        df = df[df["duration"] == duration].copy()
    return df.drop_duplicates("kic", keep="first").reset_index(drop=True)


def load_splits(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; analysis needs the original split_kics.csv."
        )
    df = pd.read_csv(path)
    if not {"kic", "split"}.issubset(df.columns):
        raise ValueError("split_kics.csv must contain kic and split columns.")
    df["kic"] = pd.to_numeric(df["kic"], errors="raise").astype(np.int64)
    if df["kic"].duplicated().any():
        raise ValueError("split_kics.csv has duplicate KIC values.")
    return df


def completed_run_problem(run_dir):
    required_files = (
        "metadata.json",
        "metrics.json",
        "config.yaml",
        "split_kics.csv",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        return f"missing {', '.join(missing)}"

    try:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        config = read_yaml(run_dir / "config.yaml")
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        return f"invalid run metadata: {exc}"

    test_metrics = metrics.get("test_reconstruction")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metrics, dict)
        or not isinstance(test_metrics, dict)
        or not {"mse", "rmse", "mae"}.issubset(test_metrics)
        or metadata.get("best_epoch") is None
        or metadata.get("best_validation_loss") is None
    ):
        return "missing completed training or test-evaluation metrics"

    autoencoder_config = config.get("autoencoder", {})
    checkpoint_candidates = [
        Path(metadata.get("checkpoint_path") or ""),
        run_dir / Path(
            autoencoder_config.get("checkpoint_name", "best_model.pth")
        ).name,
        run_dir / "best_model.pth",
    ]
    if not any(path.is_file() for path in checkpoint_candidates):
        return "missing model checkpoint"

    return None


def select_latest_completed_run(results_dir):
    completed_runs = []
    skipped_runs = []
    for run_dir in sorted(results_dir.glob("autoencoder_*")):
        if not run_dir.is_dir():
            continue
        problem = completed_run_problem(run_dir)
        if problem is None:
            completed_runs.append(run_dir)
        else:
            skipped_runs.append((run_dir, problem))

    for run_dir, problem in skipped_runs:
        print(f"Skipping incomplete autoencoder run {run_dir}: {problem}")

    if not completed_runs:
        raise FileNotFoundError(
            f"No completed autoencoder runs found in {results_dir}."
        )

    return completed_runs[-1]


def export_results_analysis():
    RESULTS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_records = []
    split_records = []

    for run_dir in sorted(DEFAULT_RESULTS_DIR.glob("autoencoder_*")):
        if not run_dir.is_dir() or completed_run_problem(run_dir) is not None:
            continue

        metadata = json.loads((run_dir / "metadata.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        test_metrics = metrics["test_reconstruction"]
        baseline_metrics = metrics.get("constant_mean_baseline", {})
        latent_metrics_path = (
            run_dir / "latent_analysis" / "latent_analysis_metrics.json"
        )
        latent_metrics = (
            json.loads(latent_metrics_path.read_text())
            if latent_metrics_path.is_file()
            else {}
        )
        run_records.append({
            "experiment_name": metadata.get("experiment_name", run_dir.name),
            "run_dir": str(run_dir),
            "best_epoch": metadata.get("best_epoch"),
            "best_validation_mse": metadata.get("best_validation_loss"),
            "test_mse": test_metrics.get("mse"),
            "test_rmse": test_metrics.get("rmse"),
            "test_mae": test_metrics.get("mae"),
            "baseline_mse": baseline_metrics.get("mse"),
            "baseline_rmse": baseline_metrics.get("rmse"),
            "baseline_mae": baseline_metrics.get("mae"),
            "normalized_test_rmse": test_metrics.get(
                "normalized_rmse_by_target_std"
            ),
            "n_test_stars": test_metrics.get("n_stars"),
            "latent_analysis_complete": bool(latent_metrics),
            "latent_reconstruction_mse": latent_metrics.get(
                "reconstruction_mse"
            ),
            "latent_reconstruction_rmse": latent_metrics.get(
                "reconstruction_rmse"
            ),
            "latent_reconstruction_mae": latent_metrics.get(
                "reconstruction_mae"
            ),
        })

        latent_vectors_path = run_dir / "latent_analysis" / "latent_vectors.csv"
        if latent_vectors_path.is_file():
            latent_vectors = pd.read_csv(latent_vectors_path)
            for split, split_df in latent_vectors.groupby("split", sort=True):
                split_records.append({
                    "experiment_name": metadata.get(
                        "experiment_name", run_dir.name
                    ),
                    "run_dir": str(run_dir),
                    "split": split,
                    "n_stars": len(split_df),
                    "mean_reconstruction_mse": split_df[
                        "reconstruction_mse"
                    ].mean(),
                    "median_reconstruction_mse": split_df[
                        "reconstruction_mse"
                    ].median(),
                    "mean_reconstruction_rmse": split_df[
                        "reconstruction_rmse"
                    ].mean(),
                    "median_reconstruction_rmse": split_df[
                        "reconstruction_rmse"
                    ].median(),
                    "mean_reconstruction_mae": split_df[
                        "reconstruction_mae"
                    ].mean(),
                    "median_reconstruction_mae": split_df[
                        "reconstruction_mae"
                    ].median(),
                })

    runs_df = pd.DataFrame(run_records)
    runs_path = RESULTS_ANALYSIS_DIR / "autoencoder_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"Saved {runs_path}")

    split_path = RESULTS_ANALYSIS_DIR / "autoencoder_split_metrics.csv"
    pd.DataFrame(split_records).to_csv(split_path, index=False)
    print(f"Saved {split_path}")

    if not runs_df.empty and runs_df[
        ["test_mse", "baseline_mse"]
    ].notna().all(axis=None):
        labels = runs_df["experiment_name"]
        positions = np.arange(len(runs_df))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(runs_df) * 1.4), 5))
        ax.bar(
            positions - width / 2,
            runs_df["test_mse"],
            width,
            label="Autoencoder test MSE",
        )
        ax.bar(
            positions + width / 2,
            runs_df["baseline_mse"],
            width,
            label="Constant-mean baseline MSE",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean squared reconstruction error")
        ax.set_title("Autoencoder test reconstruction vs. baseline")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plot_path = RESULTS_ANALYSIS_PLOTS_DIR / "autoencoder_test_vs_baseline.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved {plot_path}")


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    all_kics, all_embeddings, all_targets, all_predictions = [], [], [], []
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("Dataset must return (input, target, kic).")
        x, target, kics = batch[:3]
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        reconstruction, embedding = model(x)
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"Reconstruction/target mismatch: {tuple(reconstruction.shape)} "
                f"vs {tuple(target.shape)}"
            )
        emb = embedding.detach().cpu().numpy().reshape(x.shape[0], -1)
        if torch.is_tensor(kics):
            kics = kics.detach().cpu().numpy().reshape(-1).tolist()
        else:
            kics = np.asarray(kics).reshape(-1).tolist()
        all_kics.extend([int(k) for k in kics])
        all_embeddings.append(emb)
        all_targets.append(target.detach().cpu().numpy())
        all_predictions.append(reconstruction.detach().cpu().numpy())
    if not all_kics:
        raise ValueError("No samples were processed.")
    return (
        np.asarray(all_kics, dtype=np.int64),
        np.concatenate(all_embeddings),
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
    )


def plot_pca(embedding, metadata, output_dir, max_points):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(embedding)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    coords_df = metadata[["kic", "split"]].copy()
    coords_df["PC1"], coords_df["PC2"] = coords[:, 0], coords[:, 1]
    coords_df.to_csv(output_dir / "latent_pca_coordinates.csv", index=False)

    rng = np.random.default_rng(42)
    idx = np.arange(len(coords_df))
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    colour_fields = [
        ("evolstate", "Evolutionary state"),
        ("class", "Class"),
        ("nu_max", r"$\nu_{\max}$"),
        ("delta_nu", r"$\Delta\nu$"),
        ("metallicity", "Metallicity"),
        ("split", "Split"),
    ]
    plotted = False
    for field, label in colour_fields:
        if field not in metadata:
            continue
        values = metadata[field].iloc[idx]
        if values.notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        x, y = coords_df["PC1"].iloc[idx], coords_df["PC2"].iloc[idx]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() >= max(3, int(0.8 * len(values))):
            points = ax.scatter(x, y, c=numeric, s=9, alpha=0.7)
            fig.colorbar(points, ax=ax, label=label)
        else:
            cats = values.fillna("missing").astype(str)
            for cat in sorted(cats.unique()):
                mask = cats.to_numpy() == cat
                ax.scatter(x.to_numpy()[mask], y.to_numpy()[mask],
                           s=9, alpha=0.7, label=cat)
            ax.legend(title=label, fontsize="small")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"Autoencoder latent space (PCA); first two PCs explain "
            f"{pca.explained_variance_ratio_.sum():.1%}"
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        filename = "".join(c if c.isalnum() or c in "_-" else "_"
                           for c in field)
        fig.savefig(output_dir / f"latent_pca_by_{filename}.png", dpi=160)
        plt.close(fig)
        plotted = True
    if not plotted:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords_df["PC1"].iloc[idx], coords_df["PC2"].iloc[idx],
                   s=9, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Autoencoder latent space (PCA)")
        fig.tight_layout()
        fig.savefig(output_dir / "latent_pca.png", dpi=160)
        plt.close(fig)


def plot_umap(embedding, metadata, output_dir, max_points):
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("umap-learn unavailable; skipping UMAP.")
        return
    rng = np.random.default_rng(42)
    idx = np.arange(len(embedding))
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    z = StandardScaler().fit_transform(embedding[idx])
    coords = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                       random_state=42).fit_transform(z)
    out = metadata.iloc[idx][["kic", "split"]].copy()
    out["UMAP1"], out["UMAP2"] = coords[:, 0], coords[:, 1]
    out.to_csv(output_dir / "latent_umap_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(out["UMAP1"], out["UMAP2"], s=9, alpha=0.7)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Autoencoder latent space (UMAP)")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_umap.png", dpi=160)
    plt.close(fig)


def plot_reconstructions(kics, targets, predictions, split, output_dir, count):
    folder = output_dir / "reconstruction_examples"
    folder.mkdir(parents=True, exist_ok=True)
    test_indices = np.flatnonzero(split.astype(str).to_numpy() == "test")[:count]
    for i in test_indices:
        target = np.asarray(targets[i]).reshape(-1)
        pred = np.asarray(predictions[i]).reshape(-1)
        residual = target - pred
        fig, axes = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )
        axes[0].plot(target, label="Target", linewidth=1)
        axes[0].plot(pred, label="Reconstruction", linewidth=1, alpha=0.85)
        axes[0].set_ylabel("Signal")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].plot(residual, linewidth=0.8)
        axes[1].axhline(0, linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Reconstruction output index")
        axes[1].set_ylabel("Target − reconstruction")
        axes[1].grid(alpha=0.2)
        fig.suptitle(f"KIC {int(kics[i])} | test split")
        fig.tight_layout()
        fig.savefig(folder / f"kic_{int(kics[i])}_reconstruction.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None,
                        help="Optional specific completed autoencoder run.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing timestamped autoencoder result folders.",
    )
    parser.add_argument("--manifest", default=None,
                        help="Optional explicit manifest path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--reconstruction-examples", type=int, default=8)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        problem = completed_run_problem(run_dir)
        if problem is not None:
            raise ValueError(f"Run is not complete ({problem}): {run_dir}")
    else:
        results_dir = Path(args.results_dir).expanduser().resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Autoencoder results directory not found: {results_dir}"
            )
        run_dir = select_latest_completed_run(results_dir)
        print(f"Selected latest completed autoencoder run: {run_dir}")

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    config_path = run_dir / "config.yaml"
    split_path = run_dir / "split_kics.csv"
    metadata_path = run_dir / "metadata.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")

    config = read_yaml(config_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    manifest_path = resolve_manifest(config, run_dir, metadata, args.manifest)
    manifest = load_manifest(manifest_path, config)
    splits = load_splits(split_path)

    manifest_kics = set(manifest["kic"])
    missing = set(splits["kic"]) - manifest_kics
    if missing:
        raise ValueError(
            f"{len(missing)} KICs from the saved split are absent from the "
            "filtered manifest. Check that the manifest has not changed."
        )

    # Preserve the exact split file order; the deterministic dataset is not shuffled.
    ordered = splits["kic"].tolist()
    df = manifest.set_index("kic").loc[ordered].reset_index()

    acfg = config.get("autoencoder", {})
    input_length = int(acfg.get("input_length", 65000))
    reconstruction_length = int(acfg["reconstruction_length"])
    hidden_size = int(acfg["hidden_size"])
    num_layers = int(acfg["num_layers"])
    batch_size = args.batch_size or int(acfg.get("batch_size", 16))
    num_workers = (int(acfg.get("num_workers", 0)) if args.num_workers is None
                   else args.num_workers)
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative.")

    checkpoint_path = Path(metadata.get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / Path(
            acfg.get("checkpoint_name", "best_model.pth")
        ).name
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMAutoencoder(
        hidden_size=hidden_size,
        num_layers=num_layers,
        reconstruction_length=reconstruction_length,
    ).to(device)
    model.load_state_dict(load_checkpoint(checkpoint_path, device), strict=True)
    model.eval()

    dataset = AstroReconstructionDataset(
        df,
        target_length=input_length,
        reconstruction_length=reconstruction_length,
        deterministic=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    kics, embeddings, targets, predictions = extract(model, loader, device)
    if len(kics) != len(df) or set(kics.tolist()) != set(df["kic"].tolist()):
        raise ValueError(
            "Dataset-returned KICs do not match the selected manifest KICs."
        )
    if len(set(kics.tolist())) != len(kics):
        raise ValueError("Dataset returned duplicate KICs.")

    output_dir = run_dir / "latent_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    metrics_df = pd.DataFrame({
        "kic": kics,
        "reconstruction_mse": np.mean(errors ** 2, axis=tuple(range(1, errors.ndim))),
        "reconstruction_rmse": np.sqrt(np.mean(errors ** 2, axis=tuple(range(1, errors.ndim)))),
        "reconstruction_mae": np.mean(np.abs(errors), axis=tuple(range(1, errors.ndim))),
    })
    latent_df = pd.DataFrame(
        embeddings,
        columns=[f"latent_{i:04d}" for i in range(embeddings.shape[1])]
    )
    latent_df.insert(0, "kic", kics)
    result = metrics_df.merge(latent_df, on="kic", validate="one_to_one")
    result = result.merge(splits, on="kic", how="left", validate="one_to_one")
    extra_cols = [c for c in manifest.columns if c != "kic" and c not in result.columns]
    if extra_cols:
        result = result.merge(
            manifest[["kic"] + extra_cols], on="kic", how="left",
            validate="one_to_one"
        )
    result.to_csv(output_dir / "latent_vectors.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path.resolve()),
        "manifest": str(manifest_path),
        "n_stars": int(len(kics)),
        "embedding_dim": int(embeddings.shape[1]),
        "reconstruction_mse": float(np.mean(errors ** 2)),
        "reconstruction_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "reconstruction_mae": float(np.mean(np.abs(errors))),
        "device": str(device),
        "input_length": input_length,
        "reconstruction_length": reconstruction_length,
    }
    (output_dir / "latent_analysis_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_pca(embeddings, result, output_dir, args.max_points)
    if args.umap:
        plot_umap(embeddings, result, output_dir, args.max_points)
    plot_reconstructions(
        kics, targets, predictions, result["split"],
        output_dir, args.reconstruction_examples
    )
    export_results_analysis()

    print("Latent analysis complete.")
    print(f"Device: {device}")
    print(f"Stars: {len(kics)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Outputs: {output_dir}")
    print(f"Latent vectors: {output_dir / 'latent_vectors.csv'}")


if __name__ == "__main__":
    main()
