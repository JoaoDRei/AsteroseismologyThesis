import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

from utils import AstroBaselineDataset
from models import SimpleMLP, SimpleCNN, SimpleTransformer


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
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# -----------------------------
# Load dataset
# -----------------------------
def load_data(config):
    import pandas as pd
    synthetic = config.get("debug", {}).get("synthetic", False)

    if synthetic:
        from utils.synthetic_dataset import SyntheticPSDPeakDataset
        dataset = SyntheticPSDPeakDataset(n_samples=2000, length=config['data']['target_length'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        import pandas as pd
        df = pd.DataFrame({'kic': list(range(len(dataset))), 'nu_max': [np.nan] * len(dataset)})
        return dataset, loader, df

    df = pd.read_csv(config["data"]["manifest_path"])

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

    dataset = AstroBaselineDataset(
        df,
        mode=config["data"]["mode"],
        target_length=config["data"]["target_length"],
        deterministic=True
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataset, loader, df


def load_from_kic(kic, config, target_length, mode):
    kic_str = str(int(kic)).zfill(9)
    duration = config["data"]["filter_duration"]

    if mode == "psd":
        path = os.path.join(
            "./processed_ml_data_APOKASC3",
            f"kplr{kic_str}_{duration}_psd.npy"
        )

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
        path = os.path.join(
            "./processed_ml_data_APOKASC3",
            f"kplr{kic_str}_{duration}_clean.npy"
        )

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

# -----------------------------
# Attribution (Integrated Gradients)
# -----------------------------
def compute_ig(model, x, target_idx, baseline):
    ig = IntegratedGradients(model)

    x = x.clone().detach().requires_grad_(True)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Baseline/config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)

    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mode= config["data"]["mode"]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded config: {loaded_config_path}")

    # 2. Load data
    dataset, loader, df = load_data(config)

    sample_x, _, _ = dataset[0]
    input_length = sample_x.shape[-1]

    # 3. Model
    model = build_model(config, input_length)
    model.to(device)

    def resolve_relative_path(config_path, path):
        if os.path.isabs(path):
            return path
        cwd_path = os.path.normpath(os.path.join(os.getcwd(), path))
        if os.path.exists(cwd_path):
            return cwd_path
        base_dir = os.path.dirname(os.path.abspath(config_path))
        config_path_candidate = os.path.normpath(os.path.join(base_dir, path))
        if os.path.exists(config_path_candidate):
            return config_path_candidate
        return cwd_path

    checkpoint_dir = config['paths']['checkpoint_dir']
    checkpoint_dir = resolve_relative_path(loaded_config_path, checkpoint_dir)
    checkpoint_name = config['paths'].get('checkpoint_name')
    if checkpoint_name is None:
        target_key = "_".join(config['model'].get('targets', []))
        checkpoint_name = f"best_model_{config['model']['type']}_{target_key}.pth"
    ckpt_path = os.path.normpath(os.path.join(checkpoint_dir, checkpoint_name))
    print("Loading checkpoint from:", ckpt_path)
    if not os.path.exists(ckpt_path):
        fallback_path = os.path.normpath(os.path.join(checkpoint_dir, "best_model.pth"))
        if os.path.exists(fallback_path):
            print(f"⚠️ Specific checkpoint not found, falling back to legacy checkpoint: {fallback_path}")
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}.\n"
                f"Loaded config: {loaded_config_path}\n"
                f"checkpoint_dir: {checkpoint_dir}\n"
                f"checkpoint_name: {checkpoint_name}\n"
                f"Make sure the synthetic config is being used when debug.synthetic=true."
            )

    ckpt = torch.load(ckpt_path, map_location=device)
    print("Final layer weight shape:", ckpt["regressor.4.weight"].shape)
    model.load_state_dict(ckpt)
    model.eval()
    mode = config["data"]["mode"]
    print(f"Loaded model from {ckpt_path}")

    # 4. Output folder
    out_dir = "./Baseline/results/interpretability"
    os.makedirs(out_dir, exist_ok=True)

    # 5. Baseline
    #baseline = torch.zeros_like(sample_x).unsqueeze(0).to(device)
    #baseline = sample_x.mean(dim=-1, keepdim=True).unsqueeze(0).to(device)
    if mode == "lc":
        baseline = sample_x.mean(dim=-1, keepdim=True).unsqueeze(0).to(device)
    else:
        baseline = torch.zeros_like(sample_x).unsqueeze(0).to(device)
    # 6. Targets
    targets = config["model"]["targets"]
    target_map = {t: i for i, t in enumerate(targets)}

    # 7. Run
    n_samples = 10

    synthetic = config.get("debug", {}).get("synthetic", False)

    for i, (x, y, kic) in enumerate(loader):
        if i >= n_samples:
            break
        x = x.to(device)

        target_length = x.shape[-1]

        # ----------------------------
        # SYNTHETIC MODE (SKIP KIC)
        # ----------------------------
        if synthetic:
            # dataset already returns the signal → no file loading needed
            freq = np.linspace(0, 500, target_length)  # or whatever range you used
            power = x[0].detach().cpu().numpy().squeeze()

        # ----------------------------
        # REAL ASTRO MODE
        # ----------------------------
        else:
            
            if mode == "psd":
                freq, power = load_from_kic(
                    kic[0], config, target_length, mode=mode
                )
            else:
                time, flux = load_from_kic(
                    kic[0], config, target_length, mode=mode
                )
        # --- Forward pass ---
        with torch.no_grad():
            pred = model(x)

        for target_name, target_idx in target_map.items():

            interp_cfg = config.get("interpretability", {})
            method = interp_cfg.get("method", "ig")

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

            # --- remove batch dimension ---
            attr = attr[0]

            # --- remove channel dimension if CNN ---
            if attr.ndim == 2:
                attr = attr[0]

            save_path = os.path.join(
                out_dir,
                f"kic_{kic[0]}_{target_name}_ig.png"
            )

            kic_value = kic[0]
            if torch.is_tensor(kic_value):
                kic_value = kic_value.item()
            if isinstance(kic_value, (np.integer, np.floating)):
                kic_value = int(kic_value)
            elif isinstance(kic_value, str) and kic_value.isdigit():
                kic_value = int(kic_value)

            # For synthetic data we take nu_max from the label `y`
            if synthetic:
                nu_max = float(y.cpu().numpy().squeeze())
            else:
                matches = df[df["kic"] == kic_value]
                if matches.empty:
                    matches = df[df["kic"].astype(str) == str(kic_value)]

                if matches.empty:
                    raise ValueError(
                        f"KIC {kic_value} not found in manifest {config['data']['manifest_path']}"
                    )

                row = matches.iloc[0]
                nu_max = row["nu_max"]
            
            if mode == "psd":
                x_axis, signal = freq, power
            else:
                x_axis, signal = time, flux
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
        print(f"Processed sample {i} | KIC={kic[0]}")
        print("x_axis:", x_axis.shape, "attr:", attr.shape)
    print(f"\nSaved saliency maps to: {out_dir}")


if __name__ == "__main__":
    main()