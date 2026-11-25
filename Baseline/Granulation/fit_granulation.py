#!/usr/bin/env python3
"""
Fit a two-component Harvey-like background plus white noise and a nuisance
Gaussian oscillation envelope to each physical PSD.

This is an exploratory, reproducible first-pass fitter. Its component heights
are NOT automatically identical to Corsaro et al.'s a_meso definition.
Validate the model and normalization on a small sample before interpreting
the fitted amplitudes physically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "psd_physical"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fitted_parameters"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
MIN_FREQUENCY_UHZ = 0.1
MAX_FREQUENCY_UHZ = 283.0
NU_MAX_CSV = None
MAX_STARS = 0  # 0 fits all KICs; a positive value limits the number of KICs.


def log_bin(freq, power, bins=180, min_points=4):
    edges = np.geomspace(freq.min(), freq.max(), bins + 1)
    x, y, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freq >= lo) & (freq < hi) & np.isfinite(power) & (power > 0)
        if m.sum() >= min_points:
            x.append(np.exp(np.mean(np.log(freq[m]))))
            y.append(np.median(power[m]))
            counts.append(int(m.sum()))
    return np.asarray(x), np.asarray(y), np.asarray(counts)


def model(theta, nu):
    # theta: ln(H1), ln(nu1), alpha1, ln(H2), ln(nu2), alpha2,
    #        ln(white), ln(Gaussian amplitude), ln(Gaussian centre), ln(width)
    h1, c1, a1, h2, c2, a2, white, gamp, gcen, gwidth = theta
    H1, n1 = np.exp(h1), np.exp(c1)
    H2, n2 = np.exp(h2), np.exp(c2)
    W, G = np.exp(white), np.exp(gamp)
    centre, width = np.exp(gcen), np.exp(gwidth)

    # nu and characteristic frequencies are in microhertz.
    harvey1 = H1 / (1.0 + (nu / n1) ** a1)
    harvey2 = H2 / (1.0 + (nu / n2) ** a2)
    envelope = G * np.exp(-0.5 * (np.log(nu / centre) / width) ** 2)
    return harvey1 + harvey2 + W + envelope


def fit_one(npz_path: Path, out_plot: Path, min_freq: float, max_freq: float,
            nu_max: float | None = None):
    d = np.load(npz_path)
    freq = np.asarray(d["frequency_uHz"], float)
    power = np.asarray(d["power"], float)
    good = np.isfinite(freq) & np.isfinite(power) & (freq > 0) & (power > 0)
    good &= (freq >= min_freq) & (freq <= max_freq)
    freq, power = freq[good], power[good]
    print(f"[fit_one] Starting: {npz_path.name}")
    print(f"[fit_one] Input frequency range: [{freq.min():.6g}, {freq.max():.6g}] uHz")
    print(f"[fit_one] Input power range: [{power.min():.6g}, {power.max():.6g}]")
    print(f"[fit_one] Valid points after mask: {len(freq)}")
    if len(freq) < 100:
        raise ValueError("Insufficient valid PSD points")

    xb, yb, counts = log_bin(freq, power)
    print(f"[fit_one] Log-binned into {len(xb)} bins")
    print(f"[fit_one] Binned frequency range: [{xb.min():.6g}, {xb.max():.6g}] uHz")
    print(f"[fit_one] Binned power range: [{yb.min():.6g}, {yb.max():.6g}]")
    
    if len(xb) < 30:
        raise ValueError("Insufficient populated logarithmic bins")

    fmin, fmax = xb.min(), xb.max()
    ymax = np.percentile(yb, 90)
    ymin = max(np.percentile(yb, 10) * 0.1, np.finfo(float).tiny)
    peak_i = int(np.argmax(yb))
    centre0 = float(nu_max) if nu_max and fmin < nu_max < fmax else float(xb[peak_i])
    centre0 = np.clip(centre0, fmin * 1.05, fmax / 1.05)

    # Initial characteristic frequencies are separated; optimizer may adjust.
    p0 = np.array([
        np.log(max(ymax * 0.7, ymin)), np.log(max(fmin * 3, 0.03)), 4.0,
        np.log(max(ymax * 0.3, ymin)), np.log(max(centre0 / 2, fmin * 2)), 4.0,
        np.log(max(ymin, np.finfo(float).tiny)),
        np.log(max(ymax * 0.3, ymin)), np.log(centre0), np.log(0.35)
    ])
    lower = [
        np.log(ymin * 1e-3), np.log(fmin / 10), 2.0,
        np.log(ymin * 1e-3), np.log(fmin / 10), 2.0,
        np.log(ymin * 1e-4),
        np.log(ymin * 1e-3), np.log(fmin), np.log(0.05)
    ]
    upper = [
        np.log(ymax * 1e3), np.log(fmax * 10), 6.0,
        np.log(ymax * 1e3), np.log(fmax * 10), 6.0,
        np.log(ymax * 10),
        np.log(ymax * 1e3), np.log(fmax), np.log(1.5)
    ]

    def residual(theta):
        pred = np.maximum(model(theta, xb), np.finfo(float).tiny)
        return np.log10(pred) - np.log10(yb)
    print(f"[fit_one] Starting least_squares with p0={np.array2string(p0, precision=4)}")
    result = least_squares(residual, p0, bounds=(lower, upper),
                           loss="soft_l1", f_scale=0.15, max_nfev=12000)
    print(f"[fit_one] least_squares success={result.success}, status={result.status}, message={result.message}")
    
    if not result.success and result.status <= 0:
        raise RuntimeError(result.message)

    th = result.x
    H1, n1, a1 = np.exp(th[0]), np.exp(th[1]), th[2]
    H2, n2, a2 = np.exp(th[3]), np.exp(th[4]), th[5]
    white = np.exp(th[6])
    gamp, gcen, gwidth = np.exp(th[7]), np.exp(th[8]), np.exp(th[9])
    print(f"[fit_one] Fit parameters: H1={H1:.6g}, n1={n1:.6g}, a1={a1:.6g}")
    print(f"[fit_one] Fit parameters: H2={H2:.6g}, n2={n2:.6g}, a2={a2:.6g}")
    print(f"[fit_one] white={white:.6g}, gamp={gamp:.6g}, gcen={gcen:.6g}, gwidth={gwidth:.6g}")
    # Sort components by characteristic frequency; lower frequency = longer tau.
    comps = sorted([(H1, n1, a1), (H2, n2, a2)], key=lambda z: z[1])
    (Hlong, nlong, along), (Hshort, nshort, ashort) = comps
    # tau = 1/(2*pi*nu_char); convert microhertz to Hz.
    tau_long_s = 1.0 / (2 * np.pi * nlong * 1e-6)
    tau_short_s = 1.0 / (2 * np.pi * nshort * 1e-6)

    pred = model(th, xb)
    rms_log = float(np.sqrt(np.mean((np.log10(pred) - np.log10(yb)) ** 2)))

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(freq, power, ".", ms=1.2, alpha=0.22, label="PSD")
    ax.loglog(xb, yb, "o", ms=3, label="Log-bin medians")
    dense = np.geomspace(freq.min(), freq.max(), 1200)
    ax.loglog(dense, model(th, dense), "-", lw=1.8, label="Total fit")
    ax.set(xlabel=r"Frequency ($\mu$Hz)", ylabel="PSD power (native units)",
           title=npz_path.stem)
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)

    return {
        "file": npz_path.name,
        "kic": int(npz_path.name.split("_")[0].replace("kic", "")),
        "success": bool(result.success),
        "cost": float(result.cost),
        "rms_log10_residual": rms_log,
        "long_component_height": float(Hlong),
        "long_component_nu_char_uHz": float(nlong),
        "long_component_tau_days": float(tau_long_s / 86400),
        "long_component_alpha": float(along),
        "short_component_height": float(Hshort),
        "short_component_nu_char_uHz": float(nshort),
        "short_component_tau_days": float(tau_short_s / 86400),
        "short_component_alpha": float(ashort),
        "white_noise": float(white),
        "oscillation_envelope_amplitude": float(gamp),
        "oscillation_envelope_centre_uHz": float(gcen),
        "oscillation_envelope_log_width": float(gwidth),
        "n_raw_points": int(len(freq)),
        "n_fit_bins": int(len(xb)),
        "message": str(result.message),
    }


def main():
    nu_lookup = {}
    if NU_MAX_CSV is not None:
        import pandas as pd
        tab = pd.read_csv(NU_MAX_CSV)
        required_columns = {"KIC", "nu_max"}
        if not required_columns.issubset(tab.columns):
            raise ValueError(
                f"{NU_MAX_CSV} must contain columns KIC and nu_max."
            )
        nu_lookup = dict(zip(tab["KIC"].astype(int), tab["nu_max"].astype(float)))

    if MAX_STARS < 0:
        raise ValueError("MAX_STARS must be nonnegative (0 means all KICs).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No PSD .npz files found in {INPUT_DIR}.")
    if MAX_STARS:
        selected_kics = []
        seen_kics = set()
        for fp in files:
            kic = int(fp.name.split("_")[0].replace("kic", ""))
            if kic not in seen_kics:
                selected_kics.append(kic)
                seen_kics.add(kic)
                if len(selected_kics) == MAX_STARS:
                    break
        selected_kics = set(selected_kics)
        files = [
            fp for fp in files
            if int(fp.name.split("_")[0].replace("kic", "")) in selected_kics
        ]

    rows, failures = [], []
    for i, fp in enumerate(files, 1):
        try:
            kic = int(fp.name.split("_")[0].replace("kic", ""))
            row = fit_one(fp, PLOTS_DIR / f"{fp.stem}.png",
                          MIN_FREQUENCY_UHZ, MAX_FREQUENCY_UHZ,
                          nu_lookup.get(kic))
            rows.append(row)
        except Exception as exc:
            print(f"[main] FAILED: {fp.name} -> {exc}")
            failures.append({"file": fp.name, "error": repr(exc)})
        if i % 100 == 0:
            print(f"{i}/{len(files)} processed; successful fits: {len(rows)}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fit_results.csv", index=False)
    pd.DataFrame(failures).to_csv(OUTPUT_DIR / "fit_failures.csv", index=False)
    print(f"[main] Finished. Successful fits: {len(rows)}; failed fits: {len(failures)}")
    print(f"[main] Results: {OUTPUT_DIR}")
    print(f"[main] Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
