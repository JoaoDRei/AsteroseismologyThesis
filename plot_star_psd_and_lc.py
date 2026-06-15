from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "processed_ml_data_APOKASC3"
OUT_DIR = BASE_DIR / "plots" / "star_examples"
DEFAULT_STAR = "1429505"
DEFAULT_DURATION = "20d"


def main() -> None:
    kic = str(int(DEFAULT_STAR)).zfill(9)
    duration = DEFAULT_DURATION
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lc_path = PROC_DIR / f"kplr{kic}_{duration}_clean.npy"
    psd_path = PROC_DIR / f"kplr{kic}_{duration}_psd.npy"

    if not lc_path.exists():
        raise FileNotFoundError(f"Light curve file not found: {lc_path}")
    if not psd_path.exists():
        raise FileNotFoundError(f"PSD file not found: {psd_path}")

    time, flux = np.load(lc_path, allow_pickle=True)
    freq, power = np.load(psd_path, allow_pickle=True)

    # The processed Kepler files store cadence numbers around 55,000.
    # Display them as the more familiar 5000-style range for this plot.
    time_shifted = time - 50000.0

    # Light curve plot
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.plot(time_shifted, flux, lw=0.8, color="tab:blue")
    ax.set_xlabel("Julian Date (2450000+) (days)")
    ax.set_ylabel("Standardized flux")
    ax.set_title(f"KIC {int(kic)} — original light curve ({duration})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    lc_out = out_dir / f"kic_{kic}_{duration}_light_curve.png"
    fig.savefig(lc_out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PSD plot in log-log scale
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.loglog(freq, power, lw=0.8, color="tab:orange")
    ax.set_xlabel("Frequency (µHz)")
    ax.set_ylabel("Power")
    ax.set_title(f"KIC {int(kic)} — power spectrum ({duration})")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    psd_out = out_dir / f"kic_{kic}_{duration}_psd_loglog.png"
    fig.savefig(psd_out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:")
    print(f"  {lc_out}")
    print(f"  {psd_out}")


if __name__ == "__main__":
    main()
