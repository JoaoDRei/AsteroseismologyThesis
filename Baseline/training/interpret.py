
import argparse
import json
import os
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

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


# ============================================================
# Utilities
# ============================================================

def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def duration_in_days(value):
    normalized = str(value).strip().lower()
    if normalized.endswith("d"):
        normalized = normalized[:-1].strip()

    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid data.filter_duration {value!r}; expected a number of days "
            "such as 80 or '80d'."
        ) from exc


def normalize_kic(value):
    if torch.is_tensor(value):
        value = value.item()

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def resolve_path(path, base_dir=None):
    path = os.path.expanduser(os.fspath(path))

    if os.path.isabs(path):
        return os.path.abspath(path)

    project_path = os.path.abspath(
        os.path.join(PROJECT_ROOT, path)
    )

    if os.path.exists(project_path):
        return project_path

    cwd_path = os.path.abspath(path)

    if os.path.exists(cwd_path):
        return cwd_path

    if base_dir is not None:
        return os.path.abspath(
            os.path.join(base_dir, path)
        )

    return cwd_path


# ============================================================
# Model factory
# ============================================================

def build_model(config, input_length):
    model_type = config["model"]["type"]
    targets = config["model"].get("targets", ["nu_max"])
    output_dim = len(targets)

    if model_type == "cnn":
        return SimpleCNN(output_dim=output_dim)

    if model_type == "mlp":
        return SimpleMLP(
            input_size=input_length,
            output_dim=output_dim,
        )

    if model_type == "transformer":
        return SimpleTransformer(
            seq_length=config["data"]["seq_length"],
            output_dim=output_dim,
        )

    if model_type == "cnn_lstm":
        return CNNLSTM(
            output_dim=output_dim,
            hidden_size=config["model"].get("hidden_size", 128),
            num_layers=config["model"].get("num_layers", 2),
        )

    if model_type == "resnet1d":
        return ResNet1D(output_dim=output_dim)

    if model_type == "astroconformer_v2":
        return AstroConformerV2(
            encoder_dim=config["model"]["encoder_dim"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            patch_size=config["model"]["patch_size"],
            conv_kernel_size=config["model"]["conv_kernel_size"],
            dropout=config["model"]["dropout"],
            output_dim=output_dim,
        )

    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# Dataset
# ============================================================

def load_data(config, config_path, kics=None):
    synthetic = config.get("debug", {}).get("synthetic", False)

    if synthetic:
        if kics:
            raise ValueError(
                "--kics is not supported for synthetic data."
            )

        from Baseline.utils.synthetic_dataset import (
            SyntheticPSDPeakDataset,
        )

        dataset = SyntheticPSDPeakDataset(
            n_samples=2000,
            length=config["data"]["target_length"],
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        df = pd.DataFrame({
            "kic": list(range(len(dataset))),
            "nu_max": [np.nan] * len(dataset),
        })

        return dataset, loader, df

    manifest_path = resolve_path(
        config["data"]["manifest_path"],
        os.path.dirname(os.path.abspath(config_path)),
    )

    df = pd.read_csv(manifest_path)

    if "class" in config["model"]["targets"]:

        def normalize_ev(value):
            if pd.isna(value):
                return None

            value = str(value).strip().upper()
            return value if value in ("RGB", "RC") else None

        df["evolstate_norm"] = df["evolstate"].apply(
            normalize_ev
        )

        df = df[
            df["evolstate_norm"].notnull()
        ].reset_index(drop=True)

    filter_duration = config["data"].get("filter_duration")

    if filter_duration:
        duration_days = duration_in_days(filter_duration)
        manifest_durations = (
            df["duration"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"d$", "", regex=True)
        )
        df = df[
            pd.to_numeric(
                manifest_durations, errors="coerce"
            ) == duration_days
        ].reset_index(drop=True)

    if kics:
        requested = {normalize_kic(kic) for kic in kics}

        df_kics = df["kic"].map(normalize_kic)

        df = df[
            df_kics.isin(requested)
        ].reset_index(drop=True)

        available = set(df["kic"].map(normalize_kic))
        missing = requested - available

        if missing:
            raise ValueError(
                "Requested KICs absent after filtering: "
                + ", ".join(sorted(missing))
            )

    if df.empty:
        raise ValueError(
            "No samples remain after interpretation filtering."
        )

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
        deterministic=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    return dataset, loader, df


# ============================================================
# Reconstruct plotting axis and signal
# ============================================================

def load_from_manifest_row(row, target_length, mode):

    if mode == "psd":
        data = np.load(row["psd_path"])

        # Preserve the source dtype so interpolation matches
        # AstroBaselineDataset exactly for repeated PSD bins.
        freq = np.asarray(data[0])
        power = np.asarray(data[1])

        mask = (freq >= 0.1) & (freq <= 283)

        freq = freq[mask]
        power = power[mask]

        if len(freq) < 10:
            raise ValueError(
                "Too few frequency points after masking."
            )

        order = np.argsort(freq)
        freq = freq[order]
        power = power[order]

        if not np.isfinite(freq).all() or not np.isfinite(power).all():
            raise ValueError(
                f"PSD contains non-finite values in {row['psd_path']}."
            )

        # Same transformation as the supplied original script.
        power = np.log10(power + 1e-8)

        power = (
            (power - np.mean(power))
            / (np.std(power) + 1e-8)
        )

        log_freq = np.log10(freq + 1e-8)

        log_freq_new = np.linspace(
            log_freq.min(),
            log_freq.max(),
            target_length,
        )

        power_interp = np.interp(
            log_freq_new,
            log_freq,
            power,
        )

        freq_interp = 10 ** log_freq_new

        return freq_interp, power_interp

    if mode == "lc":
        data = np.load(row["lc_path"])

        time = np.asarray(data[0])
        flux = np.asarray(data[1])

        if len(flux) == 0:
            raise ValueError("Empty light curve.")

        if len(flux) > target_length:
            start = (len(flux) - target_length) // 2

            flux_final = flux[
                start:start + target_length
            ]

            time_final = time[
                start:start + target_length
            ]

        else:
            pad_len = target_length - len(flux)

            flux_final = np.pad(
                flux,
                (0, pad_len),
                mode="constant",
            )

            time_final = np.pad(
                time,
                (0, pad_len),
                mode="edge",
            )

        return time_final, flux_final

    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Attribution methods
# ============================================================

def _attribution_context(model, inputs):
    has_cuda_rnn = (
        inputs.is_cuda
        and any(
            isinstance(module, torch.nn.RNNBase)
            for module in model.modules()
        )
    )

    if has_cuda_rnn:
        return torch.backends.cudnn.flags(enabled=False)

    return nullcontext()


def make_baseline(x, mode):
    if mode == "lc":
        baseline = x.mean(
            dim=-1,
            keepdim=True,
        ).expand_as(x)

    elif mode == "psd":
        baseline = torch.zeros_like(x)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return baseline.detach()


def compute_ig(
    model,
    x,
    target_idx,
    baseline,
    n_steps=50,
):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)

    x = x.clone().detach().requires_grad_(True)

    with _attribution_context(model, x):
        attr = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=target_idx,
            n_steps=n_steps,
        )

    return attr.detach()


def compute_smoothgrad(
    model,
    x,
    target_idx,
    n_samples=50,
    noise_std=0.1,
):
    model.eval()
    x = x.clone().detach()

    gradients = []

    with _attribution_context(model, x):
        for _ in range(n_samples):
            noisy_x = (
                x + torch.randn_like(x) * noise_std
            ).requires_grad_(True)

            output = model(noisy_x)
            score = output[:, target_idx].sum()

            grad = torch.autograd.grad(
                score,
                noisy_x,
            )[0]

            gradients.append(grad.detach())

    return torch.stack(
        gradients,
        dim=0,
    ).mean(dim=0)

def compute_smoothgrad_ig(
    model,
    x,
    target_idx,
    baseline,
    n_samples=10,
    noise_std=0.02,
    n_steps=50,
):
    """
    Average Integrated Gradients over noisy inputs.

    The baseline remains fixed, and signed attributions
    are averaged before any absolute values are taken.
    """
    if n_samples <= 0 or n_steps <= 0 or noise_std < 0:
        raise ValueError("Invalid SmoothGrad-IG settings.")

    accumulated = torch.zeros_like(x)

    for _ in range(n_samples):
        noisy_x = x.detach() + noise_std * torch.randn_like(x)

        attribution = compute_ig(
            model=model,
            x=noisy_x,
            target_idx=target_idx,
            baseline=baseline,
            n_steps=n_steps,
        )

        accumulated += attribution

    return accumulated / n_samples







# ============================================================
# Plotting
# ============================================================

def smooth_signal(signal, window=5):
    if window <= 1:
        return signal

    return np.convolve(
        signal,
        np.ones(window) / window,
        mode="same",
    )


def plot_saliency(
    freq,
    signal,
    attr,
    save_path,
    title,
    nu_max=None,
    mode="psd",
):
    freq = np.asarray(freq).reshape(-1)
    signal = np.asarray(signal).reshape(-1)
    attr = np.asarray(attr).reshape(-1)

    if not (
        len(freq) == len(signal) == len(attr)
    ):
        raise ValueError(
            "Plot axis, signal, and attribution lengths differ."
        )

    attr = attr / (
        np.max(np.abs(attr)) + 1e-8
    )

    window = min(
        5,
        max(3, len(attr) // 20),
    )

    if window % 2 == 0:
        window += 1

    positive = smooth_signal(
        np.clip(attr, 0, None),
        window,
    )

    negative = smooth_signal(
        np.clip(attr, None, 0),
        window,
    )

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(
        freq,
        signal,
        color="black",
        alpha=0.6,
        linewidth=0.8,
    )

    if mode == "psd":
        ax1.set_xlabel("Frequency (µHz)")
        ax1.set_xscale("log")
        ax1.set_ylabel("Normalised log PSD")

        if nu_max is not None and nu_max > 0:
            ax1.axvline(
                nu_max,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=r"True $\nu_{\max}$",
            )

    else:
        ax1.set_xlabel("Time (days)")
        ax1.set_ylabel("Brightness")

    ax2 = ax1.twinx()

    ax2.fill_between(
        freq,
        0,
        positive,
        color="red",
        alpha=0.5,
        label="Positive attribution",
    )

    ax2.fill_between(
        freq,
        0,
        negative,
        color="blue",
        alpha=0.5,
        label="Negative attribution",
    )

    ax2.set_ylabel("Normalised attribution")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate model attribution arrays and plots."
    )

    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--output-dir", type=str)

    parser.add_argument(
        "--method",
        choices=("ig", "smoothgrad", "smoothgrad_ig"),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--kics",
        help="Comma-separated KIC IDs.",
    )

    args = parser.parse_args()

    if args.max_samples <= 0:
        parser.error("--max-samples must be positive.")

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)

        config_path = os.path.join(
            run_dir,
            "config.yaml",
        )

        checkpoint_path = os.path.join(
            run_dir,
            "best_model.pth",
        )

        default_output_dir = os.path.join(
            run_dir,
            "interpretability",
        )

    else:
        config_path = os.path.abspath(
            args.config or str(
                BASELINE_DIR / "config.yaml"
            )
        )

        checkpoint_path = None

        default_output_dir = str(
            BASELINE_DIR
            / "results"
            / "interpretability"
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    with open(
        config_path,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    loaded_config_path = config_path

    synthetic = config.get(
        "debug", {}
    ).get("synthetic", False)

    if synthetic:
        synthetic_config_path = config.get(
            "debug", {}
        ).get(
            "synthetic_config",
            "./config_synthetic.yaml",
        )

        if not os.path.isabs(synthetic_config_path):
            synthetic_config_path = os.path.join(
                os.path.dirname(config_path),
                synthetic_config_path,
            )

        if (
            os.path.abspath(synthetic_config_path)
            != os.path.abspath(config_path)
        ):
            with open(
                synthetic_config_path,
                encoding="utf-8",
            ) as file:
                config = yaml.safe_load(file)

            config.setdefault(
                "debug", {}
            )["synthetic"] = True

            loaded_config_path = synthetic_config_path

    interp_cfg = config.setdefault(
        "interpretability", {}
    )

    method = (
        args.method
        or interp_cfg.get("method", "ig")
    )

    if method not in ("ig", "smoothgrad", "smoothgrad_ig"):
        raise ValueError(f"Unsupported method: {method}")

    interp_cfg["method"] = method

    requested_kics = None

    if args.kics:
        requested_kics = [
            value.strip()
            for value in args.kics.split(",")
            if value.strip()
        ]

        if not requested_kics:
            parser.error("Empty --kics argument.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Configuration: {loaded_config_path}")

    dataset, loader, df = load_data(
        config,
        loaded_config_path,
        kics=requested_kics,
    )

    sample_x, _, _ = dataset[0]

    model = build_model(
        config,
        sample_x.shape[-1],
    ).to(device)

    if checkpoint_path is None:
        paths_cfg = config.get("paths", {})

        checkpoint_dir = resolve_path(
            paths_cfg.get(
                "checkpoint_dir",
                "./results/checkpoints",
            )
        )

        checkpoint_name = paths_cfg.get(
            "checkpoint_name"
        )

        if checkpoint_name is None:
            target_key = "_".join(
                config["model"].get("targets", [])
            )

            checkpoint_name = (
                f"best_model_{config['model']['type']}_"
                f"{target_key}.pth"
            )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            checkpoint_name,
        )

        if not os.path.isfile(checkpoint_path):
            legacy_path = os.path.join(
                checkpoint_dir,
                "best_model.pth",
            )

            if os.path.isfile(legacy_path):
                checkpoint_path = legacy_path

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint_path = os.path.abspath(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    mode = config["data"]["mode"]

    synthetic = config.get(
        "debug", {}
    ).get("synthetic", False)

    out_dir = os.path.abspath(
        args.output_dir or default_output_dir
    )

    os.makedirs(out_dir, exist_ok=True)

    targets = config["model"]["targets"]

    target_map = {
        target: index
        for index, target in enumerate(targets)
    }

    # The dataset is constructed from df in row order and
    # the DataLoader uses shuffle=False.
    #
    # This avoids ambiguity when the same KIC appears with
    # multiple observation durations.
    if not synthetic and len(dataset) != len(df):
        raise ValueError(
            "Dataset length differs from filtered manifest. "
            "Cannot safely match rows by position."
        )

    processed = 0

    for i, (x, y, kic) in enumerate(loader):
        if i >= args.max_samples:
            break

        x = x.to(device)
        target_length = x.shape[-1]

        kic_value = normalize_kic(kic[0])

        baseline = make_baseline(x, mode)

        if synthetic:
            x_axis = np.linspace(
                0.1,
                500,
                target_length,
            )

            signal = (
                x[0].detach().cpu().numpy().reshape(-1)
            )

            nu_max = safe_float(
                y.detach().cpu().numpy().reshape(-1)[0]
            )

            duration = None

        else:
            row = df.iloc[i]

            expected_kic = normalize_kic(row["kic"])

            if kic_value != expected_kic:
                raise ValueError(
                    f"Dataset/manifest row mismatch at index {i}: "
                    f"{kic_value} != {expected_kic}"
                )

            x_axis, signal = load_from_manifest_row(
                row,
                target_length,
                mode,
            )

            model_signal = (
                x[0].detach().cpu().numpy().reshape(-1)
            )

            if not (
                len(x_axis)
                == len(signal)
                == len(model_signal)
            ):
                raise ValueError(
                    f"Input/axis length mismatch for KIC {kic_value}"
                )

            # Verify that the plotted signal is the same as
            # the actual tensor passed into the model.
            if not np.allclose(
                signal,
                model_signal,
                rtol=1e-4,
                atol=1e-5,
            ):
                raise ValueError(
                    f"Reconstructed signal differs from model input "
                    f"for KIC {kic_value}. Check preprocessing in "
                    "AstroBaselineDataset and load_from_manifest_row."
                )

            nu_max = safe_float(
                row.get("nu_max")
            )

            duration = safe_float(
                row.get("duration")
            )

        # Include duration in filenames to avoid overwriting
        # multiple observations of the same KIC.
        if duration is not None:
            duration_label = (
                f"{duration:g}".replace(".", "p")
            )

            sample_id = (
                f"{kic_value}_{duration_label}d"
            )

        else:
            sample_id = kic_value

        sample_id = sample_id.replace(os.sep, "_")

        for target_name, target_idx in target_map.items():

            
            if method == "smoothgrad_ig":
                settings = interp_cfg.get("smoothgrad_ig", {})

                n_samples = int(settings.get("n_samples", 10))
                noise_std = float(settings.get("noise_std", 0.02))
                n_steps = int(
                    settings.get(
                        "n_steps",
                        interp_cfg.get("ig", {}).get("n_steps", 50),
                    )
                )

                attr_tensor = compute_smoothgrad_ig(
                    model,
                    x,
                    target_idx,
                    baseline,
                    n_samples=n_samples,
                    noise_std=noise_std,
                    n_steps=n_steps,
                )

                method_settings = {
                    "n_samples": n_samples,
                    "noise_std": noise_std,
                    "n_steps": n_steps,
                }

            elif method == "smoothgrad":
                settings = interp_cfg.get(
                    "smoothgrad", {}
                )

                n_samples = int(
                    settings.get("n_samples", 50)
                )

                noise_std = float(
                    settings.get("noise_std", 0.1)
                )

                if n_samples <= 0 or noise_std < 0:
                    raise ValueError(
                        "Invalid SmoothGrad settings."
                    )

                attr_tensor = compute_smoothgrad(
                    model,
                    x,
                    target_idx,
                    n_samples=n_samples,
                    noise_std=noise_std,
                )

                method_settings = {
                    "n_samples": n_samples,
                    "noise_std": noise_std,
                }

            else:
                n_steps = int(
                    interp_cfg.get(
                        "ig", {}
                    ).get("n_steps", 50)
                )

                if n_steps <= 0:
                    raise ValueError(
                        "IG n_steps must be positive."
                    )

                attr_tensor = compute_ig(
                    model,
                    x,
                    target_idx,
                    baseline,
                    n_steps=n_steps,
                )

                method_settings = {
                    "n_steps": n_steps,
                }

            attr = (
                attr_tensor[0]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if len(attr) != target_length:
                raise ValueError(
                    f"Unexpected attribution shape for {sample_id}"
                )

            filename_base = (
                f"kic_{sample_id}_{target_name}_{method}"
            )

            attribution_path = os.path.join(
                out_dir,
                filename_base + ".npz",
            )

            plot_path = os.path.join(
                out_dir,
                filename_base + ".png",
            )

            metadata = {
                "method": method,
                "kic": kic_value,
                "sample_id": sample_id,
                "duration": duration,
                "target": target_name,
                "target_index": target_idx,
                "model_type": config["model"]["type"],
                "input_mode": mode,
                "checkpoint_path": checkpoint_path,
                "baseline_type": (
                    "zero"
                    if mode == "psd"
                    else "per-input temporal mean"
                ),
                "baseline_applies": method in ("ig", "smoothgrad_ig"),
                "nu_max": nu_max,
                "frequency_unit": (
                    "uHz" if mode == "psd" else None
                ),
                "psd_frequency_grid": (
                    "uniform_log_frequency"
                    if mode == "psd"
                    else None
                ),
                "synthetic": bool(synthetic),
                **method_settings,
            }

            np.savez_compressed(
                attribution_path,
                attribution=np.asarray(
                    attr,
                    dtype=np.float32,
                ),
                input_axis=np.asarray(
                    x_axis,
                    dtype=np.float64,
                ),
                signal=np.asarray(
                    signal,
                    dtype=np.float64,
                ),
                model_input=(
                    x[0]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                baseline=(
                    baseline[0]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                metadata=np.asarray(
                    json.dumps(metadata)
                ),
            )

            plot_saliency(
                freq=x_axis,
                signal=signal,
                attr=attr,
                save_path=plot_path,
                title=(
                    f"KIC {kic_value} | "
                    f"{target_name} | {method}"
                ),
                nu_max=nu_max,
                mode=mode,
            )

        processed += 1

        print(
            f"Processed {processed}: "
            f"KIC={kic_value}, duration={duration}"
        )

    print(
        f"\nSaved {processed} samples to: {out_dir}"
    )


if __name__ == "__main__":
    main()
