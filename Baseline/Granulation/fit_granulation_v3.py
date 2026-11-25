#!/usr/bin/env python3
"""V3 pilot: matched one-/two-Harvey model comparison for physical Kepler PSDs.

Three variants on the SAME KICs, durations and log-binned PSDs:
    one_fixed : 1 Harvey, alpha=4
    two_fixed : 2 Harvey, alpha=4
    two_free  : 2 Harvey, alpha in [2,6]

All use a Gaussian oscillation envelope fixed at catalogue nu_max and a
constant white-noise floor. Fits use V2's log10-space soft-L1 objective.

Validation uses interleaved held-out log-frequency bins (not independent
periodograms). It is a useful overfitting diagnostic, NOT a calibrated
statistical model-selection test. No unverified KADACS transfer function or
nu_max-based characteristic-frequency bounds are imposed.

Examples:
    python fit_granulation_v3.py --max-stars 300
    python fit_granulation_v3.py --max-stars 300 --n-starts 5
    python fit_granulation_v3.py --max-stars 0 --no-plot
"""
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "Baseline" / "manifest.csv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "psd_physical"
SAMPLE_SEED = 42
MODELS = ("one_fixed", "two_fixed", "two_free")
TINY = np.finfo(float).tiny
MAX_STARS= 300 # 0 for full sample

def parse_filename(path: Path):
    match = re.search(r"kic0*(\d+).*?(20|55|80)d", path.stem, flags=re.I)
    if not match:
        raise ValueError(f"Cannot identify KIC/duration from {path.name}")
    return int(match.group(1)), f"{match.group(2)}d"


def load_manifest(path: Path, tolerance=0.01):
    df = pd.read_csv(path, usecols=["kic", "duration", "nu_max"])
    df["kic"] = pd.to_numeric(df.kic, errors="raise").astype("int64")
    df["nu_max"] = pd.to_numeric(df.nu_max, errors="coerce")
    df["duration"] = df.duration.astype(str).str.strip().str.lower()
    df = df[np.isfinite(df.nu_max) & (df.nu_max > 0)].copy()
    if df.empty:
        raise ValueError("No positive finite nu_max in manifest")
    g = df.groupby("kic").nu_max
    inconsistent = g.agg(lambda x: (x.max() - x.min()) / x.median() > tolerance)
    if inconsistent.any():
        raise ValueError(f"Inconsistent catalogue nu_max: {inconsistent[inconsistent].index.tolist()[:15]}")
    return g.median().to_dict(), set(zip(df.kic, df.duration))


def log_bin(freq, power, bins=180, min_points=4):
    edges = np.geomspace(freq.min(), freq.max(), bins + 1)
    indices = np.searchsorted(edges, freq, side="right") - 1
    indices = np.clip(indices, 0, bins - 1)
    xs, ys, counts = [], [], []
    for j in range(bins):
        m = indices == j
        if int(m.sum()) >= min_points:
            xs.append(float(np.exp(np.mean(np.log(freq[m])))))
            ys.append(float(np.median(power[m])))
            counts.append(int(m.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(counts)


def harvey(nu, height, nu_char, alpha):
    return height / (1.0 + (nu / nu_char) ** alpha)


def integrated_rms(height, nu_char, alpha):
    return float(np.sqrt(height * nu_char * np.pi / (alpha * np.sin(np.pi / alpha))))


def unpack(theta, model):
    if model == "one_fixed":
        h, c, w, g, lw = theta
        comps = [(float(np.exp(h)), float(np.exp(c)), 4.0)]
    elif model == "two_fixed":
        h1, c1, h2, c2, w, g, lw = theta
        comps = [(float(np.exp(h1)), float(np.exp(c1)), 4.0),
                 (float(np.exp(h2)), float(np.exp(c2)), 4.0)]
    elif model == "two_free":
        h1, c1, a1, h2, c2, a2, w, g, lw = theta
        comps = [(float(np.exp(h1)), float(np.exp(c1)), float(a1)),
                 (float(np.exp(h2)), float(np.exp(c2)), float(a2))]
    else:
        raise ValueError(model)
    comps.sort(key=lambda c: c[1])
    return comps, float(np.exp(w)), float(np.exp(g)), float(np.exp(lw))


def evaluate(theta, nu, nu_max, model):
    comps, white, amp, width = unpack(theta, model)
    parts = [harvey(nu, *component) for component in comps]
    parts.append(np.full_like(nu, white, dtype=float))
    parts.append(amp * np.exp(-0.5 * (np.log(nu / nu_max) / width) ** 2))
    return np.sum(parts, axis=0), parts


def make_bounds(xb, yb, model):
    fmin, fmax = float(xb.min()), float(xb.max())
    ymax = max(float(np.percentile(yb, 90)), TINY * 1e20)
    ymin = max(float(np.percentile(yb, 10) * 0.1), TINY * 1e20)
    log_h_lo = np.log(ymin * 1e-3)
    log_h_hi = np.log(ymax * 1e3)
    log_c_lo, log_c_hi = np.log(fmin / 10), np.log(fmax * 10)
    tail_lo = [np.log(ymin * 1e-4), np.log(ymin * 1e-3), np.log(0.08)]
    tail_hi = [np.log(ymax * 10), np.log(ymax * 1e3), np.log(0.8)]
    if model == "one_fixed":
        lo = [log_h_lo, log_c_lo, *tail_lo]
        hi = [log_h_hi, log_c_hi, *tail_hi]
    elif model == "two_fixed":
        lo = [log_h_lo, log_c_lo, log_h_lo, log_c_lo, *tail_lo]
        hi = [log_h_hi, log_c_hi, log_h_hi, log_c_hi, *tail_hi]
    else:
        lo = [log_h_lo, log_c_lo, 2.0, log_h_lo, log_c_lo, 2.0, *tail_lo]
        hi = [log_h_hi, log_c_hi, 6.0, log_h_hi, log_c_hi, 6.0, *tail_hi]
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def make_initial(xb, yb, nu_max, model, start, rng):
    fmin, fmax = float(xb.min()), float(xb.max())
    ymax = float(np.percentile(yb, 90))
    ymin = max(float(np.percentile(yb, 10) * 0.1), TINY * 1e20)
    if model == "one_fixed":
        freqs = [nu_max / 3]
        heights = [max(ymax * 0.8, ymin)]
    else:
        freqs = [nu_max / 20, nu_max / 3]
        heights = [max(ymax * 0.7, ymin), max(ymax * 0.3, ymin)]
    if start > 0:
        freqs = [f * np.exp(rng.normal(0, 1.0)) for f in freqs]
        heights = [h * np.exp(rng.normal(0, 1.0)) for h in heights]
    freqs = [float(np.clip(f, fmin * 1.1, fmax / 1.1)) for f in freqs]
    base = []
    for h, f in zip(heights, freqs):
        base.extend([np.log(h), np.log(f)])
        if model == "two_free":
            base.append(4.0 if start == 0 else float(rng.uniform(2.2, 5.8)))
    white = ymin if start == 0 else ymin * np.exp(rng.normal(0, 0.5))
    envelope = max(ymax * 0.3, ymin)
    if start > 0:
        envelope *= np.exp(rng.normal(0, 0.7))
    width = 0.35 if start == 0 else float(rng.uniform(0.10, 0.65))
    base.extend([np.log(white), np.log(envelope), np.log(width)])
    return np.asarray(base, dtype=float)


def log_residual(theta, x, y, nu_max, model):
    pred, _ = evaluate(theta, x, nu_max, model)
    return np.log10(np.maximum(pred, TINY)) - np.log10(y)


def fit_variant(xb, yb, nu_max, model, args, rng):
    # Interleaved bins: every fifth populated bin is held out. All variants
    # receive the identical split. Fit ALL bins again for final parameters.
    holdout = np.arange(len(xb)) % args.holdout_every == 0
    training = ~holdout
    if training.sum() < 25 or holdout.sum() < 5:
        raise ValueError("Too few train/holdout bins")
    lower, upper = make_bounds(xb, yb, model)
    candidates = []
    for start in range(args.n_starts):
        p0 = make_initial(xb, yb, nu_max, model, start, rng)
        p0 = np.clip(p0, lower + 1e-8, upper - 1e-8)
        try:
            fit = least_squares(
                log_residual, p0, args=(xb[training], yb[training], nu_max, model),
                bounds=(lower, upper), loss="soft_l1", f_scale=0.15,
                max_nfev=args.max_nfev,
            )
            if np.all(np.isfinite(fit.x)) and np.isfinite(fit.cost):
                candidates.append((float(fit.cost), start, fit))
        except (FloatingPointError, ValueError, OverflowError):
            continue
    if not candidates:
        raise RuntimeError("No finite training fit from any initialization")
    candidates.sort(key=lambda item: item[0])
    _, best_start, best_train = candidates[0]
    train_res = log_residual(best_train.x, xb[training], yb[training], nu_max, model)
    hold_res = log_residual(best_train.x, xb[holdout], yb[holdout], nu_max, model)
    # The final all-bin fit is initialized from the best training fit.
    final = least_squares(
        log_residual, best_train.x, args=(xb, yb, nu_max, model),
        bounds=(lower, upper), loss="soft_l1", f_scale=0.15,
        max_nfev=args.max_nfev,
    )
    if not np.all(np.isfinite(final.x)) or not np.isfinite(final.cost):
        raise RuntimeError("Nonfinite full-data fit")
    full_res = log_residual(final.x, xb, yb, nu_max, model)
    return {
        "fit": final,
        "n_successful_starts": len(candidates),
        "best_start": best_start,
        "train_rms_log10": float(np.sqrt(np.mean(train_res ** 2))),
        "holdout_rms_log10": float(np.sqrt(np.mean(hold_res ** 2))),
        "full_rms_log10": float(np.sqrt(np.mean(full_res ** 2))),
        "n_train_bins": int(training.sum()),
        "n_holdout_bins": int(holdout.sum()),
        "train_success": bool(best_train.success),
        "full_success": bool(final.success),
        "train_message": str(best_train.message),
        "full_message": str(final.message),
    }


def result_row(path, kic, duration, nu_max, xb, freq, model, fit_info, args):
    fit = fit_info["fit"]
    comps, white, amp, width = unpack(fit.x, model)
    row = {
        "file": path.name, "kic": kic, "duration": duration,
        "nu_max_catalogue": nu_max, "model": model,
        "success": bool(fit_info["full_success"]),
        "cost": float(fit.cost),
        "rms_log10_residual": fit_info["full_rms_log10"],
        "train_rms_log10_residual": fit_info["train_rms_log10"],
        "holdout_rms_log10_residual": fit_info["holdout_rms_log10"],
        "n_successful_starts": fit_info["n_successful_starts"],
        "best_start": fit_info["best_start"],
        "n_raw_points": len(freq), "n_fit_bins": len(xb),
        "n_train_bins": fit_info["n_train_bins"],
        "n_holdout_bins": fit_info["n_holdout_bins"],
        "white_noise": white, "oscillation_envelope_amplitude": amp,
        "oscillation_envelope_centre_uHz": nu_max,
        "oscillation_envelope_log_width": width,
        "message": fit_info["full_message"],
        "long_component_height": np.nan,
        "long_component_nu_char_uHz": np.nan,
        "long_component_tau_days": np.nan,
        "long_component_alpha": np.nan,
        "long_integrated_rms_native_sqrt_uHz": np.nan,
        "short_component_height": np.nan,
        "short_component_nu_char_uHz": np.nan,
        "short_component_tau_days": np.nan,
        "short_component_alpha": np.nan,
        "short_integrated_rms_native_sqrt_uHz": np.nan,
        "long_alpha_at_bound": False,
        "short_alpha_at_bound": False,
        "long_char_below_fit_band": False,
        "short_char_below_fit_band": False,
        "long_char_below_filter_timescale": False,
        "short_char_below_filter_timescale": False,
    }
    labels = ["long", "short"] if len(comps) == 2 else ["short"]
    # For the one-Harvey model, the single component is stored under 'short'
    # solely for compatibility. It is NOT a physically established short mode.
    for label, (height, char, alpha) in zip(labels, comps):
        row[f"{label}_component_height"] = height
        row[f"{label}_component_nu_char_uHz"] = char
        row[f"{label}_component_tau_days"] = 1 / (2 * np.pi * char * 1e-6 * 86400)
        row[f"{label}_component_alpha"] = alpha
        row[f"{label}_integrated_rms_native_sqrt_uHz"] = integrated_rms(height, char, alpha)
        row[f"{label}_alpha_at_bound"] = bool(model == "two_free" and min(abs(alpha - 2), abs(alpha - 6)) <= 0.01)
        row[f"{label}_char_below_fit_band"] = bool(char < float(xb.min()))
        # 1/T is a heuristic filter frequency scale, NOT the measured response.
        row[f"{label}_char_below_filter_timescale"] = bool(char < 1e6 / (int(duration[:-1]) * 86400))
    return row


def plot_fit(path, kic, duration, nu_max, freq, power, xb, yb, results, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    dense = np.geomspace(freq.min(), freq.max(), 1000)
    ax = axes[0, 0]
    ax.loglog(freq, power, ".", alpha=0.10, ms=1, color="0.5", label="Raw PSD")
    ax.loglog(xb, yb, ".", ms=4, color="black", label="Binned median")
    for model in MODELS:
        if model in results:
            pred, _ = evaluate(results[model]["fit"].x, dense, nu_max, model)
            ax.loglog(dense, pred, lw=1.7, label=model)
    ax.axvline(nu_max, color="0.4", ls=":", label=r"Catalogue $\nu_{\max}$")
    ax.set(title="Model comparison", ylabel="PSD (native units)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    ax2 = axes[0, 1]
    chosen = "two_fixed" if "two_fixed" in results else next(iter(results))
    pred, parts = evaluate(results[chosen]["fit"].x, dense, nu_max, chosen)
    ax2.loglog(xb, yb, ".", ms=4, color="black", label="Binned median")
    ax2.loglog(dense, pred, lw=2, label=chosen)
    names = (["Harvey 1", "Harvey 2"] if chosen != "one_fixed" else ["Harvey"])
    names += ["White", "Envelope"]
    for component, name in zip(parts, names):
        ax2.loglog(dense, component, "--", alpha=0.75, label=name)
    ax2.axvline(nu_max, color="0.4", ls=":")
    ax2.set(title=f"{chosen} decomposition", ylabel="PSD (native units)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_ylim(ax.get_ylim())
    for col in range(2):
        for model in MODELS:
            if model not in results:
                continue
            info = results[model]
            residual = log_residual(info["fit"].x, xb, yb, nu_max, model)
            axes[1, col].semilogx(xb, -residual, ".", ms=3, label=model)
        axes[1, col].axhline(0, color="black", lw=1)
        axes[1, col].set(ylabel="log10(data/fit)")
        axes[1, col].grid(True, which="both", alpha=0.2)
    for col in range(2):
        for model in MODELS:
            if model not in results:
                continue
            info = results[model]
            axes[2, col].plot([0, 1], [info["train_rms_log10"], info["holdout_rms_log10"]],
                              "o-", label=model)
        axes[2, col].set(xticks=[0, 1], xticklabels=["Train", "Held out"],
                          ylabel="log10 RMS (dex)")
        axes[2, col].grid(True, alpha=0.2)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle(
        f"KIC {kic} | {duration} | "
        f"$\\nu_{{\\max}}$={nu_max:.2f} microHz"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def save_tables(rows, failures, out):
    pd.DataFrame(rows).to_csv(out / "fit_results.csv", index=False)
    pd.DataFrame(failures, columns=["file", "kic", "duration", "model", "error"]).to_csv(
        out / "fit_failures.csv", index=False)


def write_summary(rows, out):
    df = pd.DataFrame(rows)
    if df.empty:
        (out / "model_comparison.csv").write_text("model,n,median_full_rms,median_holdout_rms\n")
        return
    stats = []
    for model, part in df.groupby("model"):
        stats.append({
            "model": model, "n": len(part),
            "median_full_rms": part.rms_log10_residual.median(),
            "median_train_rms": part.train_rms_log10_residual.median(),
            "median_holdout_rms": part.holdout_rms_log10_residual.median(),
            "long_char_below_fit_band_fraction": part.long_char_below_fit_band.mean(),
            "long_char_below_filter_timescale_fraction": part.long_char_below_filter_timescale.mean(),
            "long_alpha_at_bound_fraction": part.long_alpha_at_bound.mean(),
        })
    pd.DataFrame(stats).to_csv(out / "model_comparison.csv", index=False)
    wide = df.pivot_table(index=["kic", "duration"], columns="model",
                          values="holdout_rms_log10_residual", aggfunc="first")
    wide.to_csv(out / "paired_holdout_rms.csv")
    if set(MODELS).issubset(wide.columns):
        paired = wide.dropna(subset=list(MODELS))
        if len(paired):
            info = {
                "n_matched_spectra": len(paired),
                "fraction_two_fixed_better_than_one_fixed": float((paired.two_fixed < paired.one_fixed).mean()),
                "fraction_two_free_better_than_two_fixed": float((paired.two_free < paired.two_fixed).mean()),
                "median_paired_holdout_improvement_one_to_two_fixed_dex": float(np.median(paired.one_fixed - paired.two_fixed)),
                "median_paired_holdout_improvement_two_fixed_to_free_dex": float(np.median(paired.two_fixed - paired.two_free)),
            }
            (out / "paired_summary.json").write_text(json.dumps(info, indent=2))
            print("Matched holdout comparison:", json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--min-frequency", type=float, default=0.1)
    parser.add_argument("--max-frequency", type=float, default=283.0)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--min-bin-points", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=12000)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument(
        '--max-stars',
        type=int,
        default=MAX_STARS,
        help=f'Unique KICs; 0 means all (default: MAX_STARS={MAX_STARS})',
    )
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    if args.max_stars < 0 or args.n_starts < 1 or args.holdout_every < 3:
        parser.error("Require max-stars>=0, n-starts>=1, holdout-every>=3")
    if not (0 < args.min_frequency < args.max_frequency):
        parser.error("Require 0 < min-frequency < max-frequency")
    label = str(args.max_stars) if args.max_stars else "all"
    args.output_dir = args.output_dir or PROJECT_ROOT / "data" / f"fitted_parameters_v3_pilot_{label}"
    args.plot_dir = args.plot_dir or PROJECT_ROOT / "results" / f"granulation_v3_pilot_plots_{label}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup, pairs = load_manifest(args.manifest)
    files = []
    for path in sorted(args.input_dir.glob("*.npz")):
        try:
            kic, duration = parse_filename(path)
        except ValueError:
            continue
        if kic in lookup and (kic, duration) in pairs:
            files.append((path, kic, duration))
    if not files:
        raise ValueError("No physical PSDs match manifest")
    if args.max_stars:
        available = np.array(sorted({kic for _, kic, _ in files}), dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(available, size=min(args.max_stars, len(available)), replace=False).tolist())
        files = [item for item in files if item[1] in chosen]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"models": list(MODELS), "alpha_bounds": [2, 6],
                   "envelope_width_bounds": [0.08, 0.8],
                   "objective": "log10 median PSD soft_l1 f_scale=0.15",
                   "holdout": "every fifth populated log bin, interleaved",
                   "filter_transfer_function_applied": False,
                   "nu_char_nu_max_scaling_constraint_applied": False})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    rows, failures = [], []
    print(f"Fitting {len(files)} spectra, {len(set(k for _, k, _ in files))} KICs, {len(MODELS)} variants", flush=True)
    for i, (path, kic, duration) in enumerate(files, 1):
        try:
            with np.load(path) as data:
                freq = np.asarray(data["frequency_uHz"], dtype=float)
                power = np.asarray(data["power"], dtype=float)
            good = (np.isfinite(freq) & np.isfinite(power) &
                    (freq >= args.min_frequency) & (freq <= args.max_frequency) & (power > 0))
            freq, power = freq[good], power[good]
            if len(freq) < 100:
                raise ValueError("Fewer than 100 valid PSD samples")
            order = np.argsort(freq)
            freq, power = freq[order], power[order]
            xb, yb, _ = log_bin(freq, power, args.bins, args.min_bin_points)
            if len(xb) < 30:
                raise ValueError("Fewer than 30 populated log bins")
            nu_max = lookup[kic]
            if not (xb.min() < nu_max < xb.max()):
                raise ValueError(f"nu_max={nu_max} outside binned PSD [{xb.min()}, {xb.max()}]")
            successes = {}
            for model in MODELS:
                try:
                    # Deterministic independent start stream for each spectrum/model.
                    model_index = MODELS.index(model)
                    seed_seq = np.random.SeedSequence([args.seed, int(kic), int(duration[:-1]), model_index])
                    fit_info = fit_variant(xb, yb, nu_max, model, args, np.random.default_rng(seed_seq))
                    successes[model] = fit_info
                    rows.append(result_row(path, kic, duration, nu_max, xb, freq, model, fit_info, args))
                except Exception as exc:
                    failures.append({"file": path.name, "kic": kic, "duration": duration,
                                     "model": model, "error": repr(exc)})
            if args.plot and successes:
                try:
                    plot_fit(path, kic, duration, nu_max, freq, power, xb, yb, successes,
                             args.plot_dir / f"{path.stem}.png")
                except Exception as exc:
                    print(f"Plot warning for {path.name}: {exc}", flush=True)
        except Exception as exc:
            for model in MODELS:
                failures.append({"file": path.name, "kic": kic, "duration": duration,
                                 "model": model, "error": repr(exc)})
        if i % 25 == 0 or i == len(files):
            save_tables(rows, failures, args.output_dir)
            print(f"{i}/{len(files)} spectra | successful variant fits={len(rows)} failures={len(failures)}", flush=True)
    write_summary(rows, args.output_dir)
    print(f"Finished: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
