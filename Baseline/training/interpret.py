import os
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

from torch.utils.data import DataLoader

from Baseline.paths import BASELINE_DIR, PROJECT_ROOT
from Baseline.utils import AstroBaselineDataset
from Baseline.models import (
    AstroConformerV2,
    CNNLSTM,
    ResNet1D,
    SimpleMLP,
    SimpleCNN,
    SimpleTransformer,
)


# -----------------------------
# Model factory
# -----------------------------
def build_model(config, input_length):
    model_type = config["model"]["type"]
    targets = config["model"].get("targets", ["nu_max"])
    output_dim = len(targets)

    if model_type == "cnn":
        model = SimpleCNN(output_dim=output_dim)
    elif model_type == "mlp":
        model = SimpleMLP(input_size=input_length, output_dim=output_dim)
    elif model_type == "transformer":
        model = SimpleTransformer(seq_length=config["data"]["seq_length"], output_dim=output_dim)
    elif model_type == "cnn_lstm":
        model = CNNLSTM(
            output_dim=output_dim,
            hidden_size=config["model"].get("hidden_size", 128),
            num_layers=config["model"].get("num_layers", 2),
        )
    elif model_type == "resnet1d":
        model = ResNet1D(output_dim=output_dim)
    elif model_type == "astroconformer_v2":
        model = AstroConformerV2(
            encoder_dim=config["model"]["encoder_dim"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            patch_size=config["model"]["patch_size"],
            conv_kernel_size=config["model"]["conv_kernel_size"],
            dropout=config["model"]["dropout"],
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# -----------------------------
# Load dataset
# -----------------------------
def resolve_path(path, base_dir=None):
    path = os.path.expanduser(os.fspath(path))
    if os.path.isabs(path):
        return os.path.abspath(path)

    project_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if os.path.exists(project_path):
        return project_path

    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path

    if base_dir is not None:
        return os.path.abspath(os.path.join(base_dir, path))

    return cwd_path


def load_data(config, config_path, kics=None):
    import pandas as pd
    synthetic = config.get("debug", {}).get("synthetic", False)

    if synthetic:
        if kics:
            raise ValueError("--kics is not supported for synthetic data.")
        from Baseline.utils.synthetic_dataset import SyntheticPSDPeakDataset
        dataset = SyntheticPSDPeakDataset(n_samples=2000, length=config['data']['target_length'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        df = pd.DataFrame({'kic': list(range(len(dataset))), 'nu_max': [np.nan] * len(dataset)})
        return dataset, loader, df

    manifest_path = resolve_path(
        config["data"]["manifest_path"],
        os.path.dirname(os.path.abspath(config_path)),
    )
    df = pd.read_csv(manifest_path)

    if "class" in config["model"]["targets"]:

        def normalize_ev(v):
            if pd.isna(v):
                return None
            v = str(v).strip().upper()
            return v if v in ["RGB", "RC"] else None

        df["evolstate_norm"] = df["evolstate"].apply(normalize_ev)
        df = df[df["evolstate_norm"].notnull()].reset_index(drop=True)

    if config["data"].get("filter_duration"):
        df = df[df["duration"] == config["data"]["filter_duration"]].reset_index(drop=True)

    if kics:
        requested_kics = {str(kic) for kic in kics}
        df_kics = df["kic"].astype(str).str.replace(r"\.0$", "", regex=True)
        df = df[df_kics.isin(requested_kics)].reset_index(drop=True)
        missing_kics = requested_kics - set(df["kic"].astype(str))
        if missing_kics:
            raise ValueError(
                "Requested KICs are absent after manifest filtering: "
                + ", ".join(sorted(missing_kics))
            )

    if df.empty:
        raise ValueError("No samples remain after interpretation data filtering.")

    for column in ("lc_path", "psd_path"):
        if column in df.columns:
            df[column] = df[column].map(
                lambda value: resolve_path(
                    value,
                    os.path.dirname(manifest_path),
                )
                if isinstance(value, str)
                else value
            )

    dataset = AstroBaselineDataset(
        df,
        targets=config["model"]["targets"],
        mode=config["data"]["mode"],
        target_length=config["data"]["target_length"],
        deterministic=True
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataset, loader, df


def load_from_manifest_row(row, target_length, mode):
    if mode == "psd":
        path = row["psd_path"]
        data = np.load(path)
        freq = data[0]
        power = data[1]

        # --- SAME AS dataset.py ---

        # 1. Mask
        mask = (freq >= 0.1) & (freq <= 283)
        freq, power = freq[mask], power[mask]

        # 2. Sort
        idx = np.argsort(freq)
        freq, power = freq[idx], power[idx]

        # 3. Log power + normalize
        power = np.log10(power + 1e-8)
        power = (power - np.mean(power)) / (np.std(power) + 1e-8)

        # 4. Log-frequency interpolation (CRITICAL)
        freq_log = np.log10(freq + 1e-8)
        freq_log_new = np.linspace(freq_log.min(), freq_log.max(), target_length)

        power_interp = np.interp(freq_log_new, freq_log, power)
        freq_interp = 10 ** freq_log_new

        return freq_interp, power_interp

    elif mode == "lc":
        path = row["lc_path"]
        data = np.load(path)
        time = data[0]
        flux = data[1]

        # --- DETERMINISTIC version of dataset.py ---

        if len(flux) > target_length:
            # Instead of random crop → center crop
            start = (len(flux) - target_length) // 2
            flux_final = flux[start:start + target_length]
            time_final = time[start:start + target_length]
        else:
            # Same padding
            pad_len = target_length - len(flux)
            flux_final = np.pad(flux, (0, pad_len), mode='constant')
            time_final = np.pad(time, (0, pad_len), mode='edge')

        return time_final, flux_final
    else:
        raise ValueError(f"Unsupported input mode: {mode}")

# -----------------------------
# Attribution (Integrated Gradients)
# -----------------------------
def _attribution_context(model, inputs):
    has_cuda_rnn = inputs.is_cuda and any(
        isinstance(module, torch.nn.RNNBase) for module in model.modules()
    )
    if has_cuda_rnn:
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


def compute_ig(model, x, target_idx, baseline):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)

    x = x.clone().detach().requires_grad_(True)

    with _attribution_context(model, x):
        attr = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=target_idx
        )

    return attr


def compute_smoothgrad(model, x, target_idx, baseline, n_samples=50, noise_std=0.1):
    model.eval()

    x = x.clone().detach()

    grads = []

    with _attribution_context(model, x):
        for _ in range(n_samples):
            noisy_x = x + torch.randn_like(x) * noise_std
            noisy_x.requires_grad_(True)

            out = model(noisy_x)

            score = out[:, target_idx].sum()

            grad = torch.autograd.grad(score, noisy_x)[0]

            grads.append(grad.detach())

    grads = torch.stack(grads, dim=0)

    return grads.mean(dim=0)

# -----------------------------
# Smooth function
# -----------------------------
def smooth_signal(signal, window=15):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


# -----------------------------
# Plot function (FIXED)
# -----------------------------
def plot_saliency(freq, signal, attr, save_path, title, nu_max=None, mode="psd", config=None):

    signal = signal.squeeze()
    attr = attr.squeeze()

    # Ensure numpy arrays
    freq = np.asarray(freq)
    signal = np.asarray(signal)
    attr = np.asarray(attr)


    # --- Ensure alignment ---
    assert len(freq) == len(attr), f"Mismatch: freq={len(freq)}, attr={len(attr)}"

    # --- Normalize attribution ---
    if config["interpretability"]["method"]=="ig":
        attr = attr / (np.max(np.abs(attr)) + 1e-8)
    elif config["interpretability"]["method"]=="smoothgrad":
        #attr = attr / (np.linalg.norm(attr) + 1e-8)
        attr = attr / (np.max(np.abs(attr)) + 1e-8)


    # --- Smooth attribution ---
    # choose window not larger than the signal
    window = min(5, max(3, int(len(attr) // 20)))
    if window % 2 == 0:
        window += 1
    attr_pos = np.clip(attr, 0, None)
    attr_neg = np.clip(attr, None, 0)

    attr_pos = smooth_signal(attr_pos, window=window)
    attr_neg = smooth_signal(attr_neg, window=window)

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # --- Plot PSD ---
    ax1.plot(freq, signal, color="black", alpha=0.6)


    if mode == "psd":
        ax1.set_xlabel("Frequency (µHz)")
        ax1.set_xscale("log")
        ax1.set_ylabel("PSD Power")
    elif mode == "lc":
        ax1.set_xlabel("Time (days)")
        ax1.set_ylabel("Brightness")

    # νmax vertical line
    if nu_max is not None and mode == "psd":
        ax1.axvline(
            x=nu_max,
            color="green",
            linestyle="--",
            linewidth=2,
            label=r"$\nu_{\max}$ (true)"
        )

    # Attribution overlay
    ax2 = ax1.twinx()

    ax2.fill_between(freq, 0, attr_pos, color="red", alpha=0.5, label="Positive")
    ax2.fill_between(freq, 0, attr_neg, color="blue", alpha=0.5, label="Negative")

    ax2.set_ylabel("Attribution")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model attributions for a trained experiment."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Training run directory containing config.yaml and best_model.pth.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Config path for standalone use when --run-dir is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory in which attribution plots are written.",
    )
    parser.add_argument(
        "--method",
        choices=("ig", "smoothgrad"),
        help="Attribution method (defaults to the saved config).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of stars to process.",
    )
    parser.add_argument(
        "--kics",
        help="Optional comma-separated list of KIC IDs to process.",
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        parser.error("--max-samples must be positive.")

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        config_path = os.path.join(run_dir, "config.yaml")
        checkpoint_path = os.path.join(run_dir, "best_model.pth")
        default_output_dir = os.path.join(run_dir, "interpretability")
    else:
        config_path = os.path.abspath(
            args.config or str(BASELINE_DIR / "config.yaml")
        )
        run_dir = None
        checkpoint_path = None
        default_output_dir = str(BASELINE_DIR / "results" / "interpretability")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    mode = config["data"]["mode"]
    # If the main config only signals synthetic mode, switch to the synthetic config.
    synthetic = config.get("debug", {}).get("synthetic", False)
    loaded_config_path = config_path
    if synthetic:
        synthetic_config_path = config.get("debug", {}).get("synthetic_config", "./config_synthetic.yaml")
        base_dir = os.path.dirname(os.path.abspath(config_path))
        synthetic_config_path = os.path.join(base_dir, synthetic_config_path) if not os.path.isabs(synthetic_config_path) else synthetic_config_path
        if os.path.abspath(synthetic_config_path) != os.path.abspath(config_path):
            with open(synthetic_config_path, "r") as f:
                config = yaml.safe_load(f)
            config.setdefault("debug", {})["synthetic"] = True
            loaded_config_path = synthetic_config_path

    interpretability_config = config.setdefault("interpretability", {})
    method = args.method or interpretability_config.get("method", "ig")
    if method not in ("ig", "smoothgrad"):
        raise ValueError(f"Unsupported attribution method: {method}")
    interpretability_config["method"] = method

    requested_kics = None
    if args.kics:
        requested_kics = [
            value.strip()
            for value in args.kics.split(",")
            if value.strip()
        ]
        if not requested_kics:
            parser.error("--kics must contain at least one KIC ID.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded config: {loaded_config_path}")

    dataset, loader, df = load_data(
        config,
        loaded_config_path,
        kics=requested_kics,
    )

    sample_x, _, _ = dataset[0]
    input_length = sample_x.shape[-1]

    model = build_model(config, input_length)
    model.to(device)

    if checkpoint_path is None:
        paths_config = config.get("paths", {})
        checkpoint_dir = resolve_path(
            paths_config.get("checkpoint_dir", "./results/checkpoints")
        )
        checkpoint_name = paths_config.get("checkpoint_name")
        if checkpoint_name is None:
            target_key = "_".join(config["model"].get("targets", []))
            checkpoint_name = (
                f"best_model_{config['model']['type']}_{target_key}.pth"
            )
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        if not os.path.isfile(checkpoint_path):
            legacy_checkpoint = os.path.join(
                checkpoint_dir,
                "best_model.pth",
            )
            if os.path.isfile(legacy_checkpoint):
                checkpoint_path = legacy_checkpoint

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    mode = config["data"]["mode"]
    print(f"Loaded model from {checkpoint_path}")

    out_dir = os.path.abspath(args.output_dir or default_output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if mode == "lc":
        baseline = sample_x.mean(dim=-1, keepdim=True).unsqueeze(0).to(device)
    else:
        baseline = torch.zeros_like(sample_x).unsqueeze(0).to(device)

    targets = config["model"]["targets"]
    target_map = {t: i for i, t in enumerate(targets)}
    synthetic = config.get("debug", {}).get("synthetic", False)

    for i, (x, y, kic) in enumerate(loader):
        if i >= args.max_samples:
            break
        x = x.to(device)
        target_length = x.shape[-1]

        if synthetic:
            freq = np.linspace(0, 500, target_length)
            power = x[0].detach().cpu().numpy().squeeze()
        else:
            kic_value = kic[0].item() if torch.is_tensor(kic[0]) else kic[0]
            matches = df[df["kic"].astype(str) == str(kic_value)]
            if matches.empty:
                matches = df[
                    pd.to_numeric(df["kic"], errors="coerce") == float(kic_value)
                ]
            if matches.empty:
                raise ValueError(
                    f"KIC {kic_value} is missing from the filtered manifest."
                )
            row = matches.iloc[0]
            x_axis, signal = load_from_manifest_row(
                row,
                target_length,
                mode,
            )

        for target_name, target_idx in target_map.items():
            interp_cfg = config.get("interpretability", {})

            if method == "smoothgrad":
                sg_cfg = interp_cfg.get("smoothgrad", {})
                attr = compute_smoothgrad(
                    model,
                    x,
                    target_idx,
                    baseline,
                    n_samples=sg_cfg.get("n_samples", 50),
                    noise_std=sg_cfg.get("noise_std", 0.1), 
                )
            else:
                attr = compute_ig(model, x, target_idx, baseline)

            attr = attr.detach().cpu().numpy()
            attr = attr[0]
            if attr.ndim == 2:
                attr = attr[0]

            kic_value = (
                kic[0].item()
                if torch.is_tensor(kic[0])
                else kic[0]
            )
            filename_kic = str(kic_value).replace(os.sep, "_")
            save_path = os.path.join(
                out_dir,
                f"kic_{filename_kic}_{target_name}_{method}.png"
            )

            if synthetic:
                nu_max = float(y.cpu().numpy().squeeze())
            else:
                nu_max = row["nu_max"]

            plot_saliency(
                freq=x_axis,
                signal=signal,
                attr=attr,
                save_path=save_path,
                title=f"KIC {kic[0]} - {target_name}",
                nu_max=nu_max,
                mode=mode,
                config=config
            )
        print(f"Processed sample {i} | KIC={kic_value}")
    print(f"\nSaved saliency maps to: {out_dir}")


if __name__ == "__main__":
    main()