#!/usr/bin/env python3
"""Controlled PSD occlusion with the EXISTING trained ResNet1D model.

Run from the project root without arguments:
    python -m Baseline.analysis.controlled_occlusion

Defaults: the 80d multitask ResNet1D checkpoint, the ten IG robustness pilot
stars, V3 two_fixed fit_results.csv, and a separate two_fixed analysis directory.
All defaults can be overridden with command-line arguments. No retraining.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from Baseline.training.interpret import build_model, load_data, load_from_manifest_row, normalize_kic, safe_float
from Baseline.paths import BASELINE_DIR

EPS = 1e-8
DEFAULT_RUN_DIR = (
    BASELINE_DIR / 'results'
    / 'resnet1d_psd_80d_nu_max_delta_nu_class_20261018_190922_635749'
)
DEFAULT_FIT_CSV = (
    BASELINE_DIR.parent / 'data'
    / 'fitted_parameters_v3_two_fixed_all' / 'fit_results.csv'
)
DEFAULT_OUTPUT_DIR = (
    BASELINE_DIR / 'results_analysis' / 'controlled_occlusion_two_fixed'
)
DEFAULT_KICS = (
    '11245339,11805876,12784948,1429505,3533464,'
    '4348593,4991033,5172229,6501869,9021799'
)

def fit_parts(freq, row):
    """Reconstruct the two_fixed Harvey + white + Gaussian model in native PSD units."""
    if str(row['model']) != 'two_fixed':
        raise ValueError('Only two_fixed fit rows are supported')
    bg = np.full_like(freq, float(row['white_noise']), dtype=float)
    for prefix in ('long', 'short'):
        h = float(row[f'{prefix}_component_height'])
        c = float(row[f'{prefix}_component_nu_char_uHz'])
        a = float(row[f'{prefix}_component_alpha'])
        if not (np.isfinite(h) and np.isfinite(c) and np.isfinite(a) and h > 0 and c > 0 and a > 0):
            raise ValueError('Invalid Harvey fit parameters')
        bg += h / (1 + (freq / c)**a)
    amp = float(row['oscillation_envelope_amplitude'])
    width = float(row['oscillation_envelope_log_width'])
    center = float(row['oscillation_envelope_centre_uHz'])
    if not (amp >= 0 and width > 0 and center > 0):
        raise ValueError('Invalid Gaussian envelope parameters')
    envelope = amp * np.exp(-0.5 * (np.log(freq / center) / width)**2)
    return bg, envelope

def raw_psd(path):
    with np.load(path, allow_pickle=False) if str(path).endswith('.npz') else open(path, 'rb') as obj:
        if isinstance(obj, np.lib.npyio.NpzFile):
            if 'frequency_uHz' in obj and 'power' in obj:
                f, p = obj['frequency_uHz'], obj['power']
            else:
                arr = obj[obj.files[0]]
                f, p = arr[0], arr[1]
        else:
            arr = np.load(obj, allow_pickle=False)
            f, p = arr[0], arr[1]
    f, p = np.asarray(f, float), np.asarray(p, float)
    good = np.isfinite(f) & np.isfinite(p) & (f >= 0.1) & (f <= 283) & (p + EPS > 0)
    f, p = f[good], p[good]
    order = np.argsort(f)
    f, p = f[order], p[order]
    if len(f) < 10 or np.any(np.diff(f) <= 0):
        raise ValueError('Invalid/duplicate PSD frequencies')
    return f, p

def transform_power(power, freq, output_axis, mu, sd):
    z = (np.log10(np.maximum(power + EPS, EPS)) - mu) / (sd + EPS)
    return np.interp(np.log10(output_axis), np.log10(freq), z).astype(np.float32)

def summarize(df):
    return df.groupby(['target', 'intervention'], as_index=False).agg(
        n_stars=('kic', 'nunique'),
        median_fraction_masked=('fraction_bins_masked', 'median'),
        median_abs_delta=('abs_delta_output', 'median'),
        median_abs_delta_std=('abs_delta_in_output_std', 'median'),
        median_abs_delta_probability=('abs_delta_probability', 'median'),
        fraction_prediction_flipped=('prediction_flipped', 'mean'),
        median_delta_abs_error=('delta_abs_error', 'median'),
        mean_delta_abs_error=('delta_abs_error', 'mean'),
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument('--fit-csv', type=Path, default=DEFAULT_FIT_CSV, help='V3 two_fixed fit_results.csv')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Separate two_fixed output directory')
    ap.add_argument('--kics', type=str, default=DEFAULT_KICS, help='Comma-separated KICs; defaults to ten IG pilot stars')
    ap.add_argument('--max-stars', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--plots', type=int, default=10, help='Number of illustrative PSD plots')
    args = ap.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / 'config.yaml'
    with config_path.open() as f:
        config = yaml.safe_load(f)
    if config['data']['mode'] != 'psd':
        raise ValueError('PSD runs only')
    requested = [s.strip() for s in args.kics.split(',') if s.strip()] if args.kics is not None else None
    dataset, _, df = load_data(config, str(config_path), kics=requested)
    keys = df.kic.map(normalize_kic)
    if requested:
        selected = {normalize_kic(s.strip()) for s in requested}
    else:
        unique = np.asarray(sorted(keys.unique()))
        selected = set(np.random.default_rng(args.seed).choice(unique, size=min(args.max_stars, len(unique)), replace=False))
    indices, seen = [], set()
    for i, k in enumerate(keys):
        if k in selected and k not in seen:
            indices.append(i)
            seen.add(k)
    if not indices:
        raise ValueError('No stars selected')
    fit_lookup = {}
    if args.fit_csv:
        fits = pd.read_csv(args.fit_csv.expanduser().resolve())
        if 'model' not in fits or not (fits['model'].astype(str) == 'two_fixed').any():
            raise ValueError('The supplied fit CSV contains no two_fixed rows. Run your V3 fitter with MODELS=("two_fixed",) or supply its existing results.')
        fits = fits[(fits['model'] == 'two_fixed') & (fits['success'].astype(str).str.lower().isin(['true', '1']))]
        for _, r in fits.iterrows():
            key = (normalize_kic(r['kic']), str(r['duration']).strip().lower())
            if key in fit_lookup:
                raise ValueError(f'Duplicate fit row for {key}')
            fit_lookup[key] = r
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x0, _, _ = dataset[indices[0]]
    model = build_model(config, x0.shape[-1]).to(device)
    checkpoint = run_dir / 'best_model.pth'
    if not checkpoint.exists():
        target_key = '_'.join(config['model']['targets'])
        candidates = list(run_dir.glob(f'best_model_{config["model"]["type"]}_{target_key}.pth'))
        if len(candidates) != 1:
            raise FileNotFoundError(f'Cannot locate a unique checkpoint in {run_dir}; expected best_model.pth')
        checkpoint = candidates[0]
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get('model_state_dict', state.get('state_dict', state))
    model.load_state_dict(state)
    model.eval()
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'plots').mkdir(exist_ok=True)
    targets = list(config['model']['targets'])
    records, warnings = [], []
    diagnostics = []
    for count, i in enumerate(indices):
        row = df.iloc[i]
        x, y, k = dataset[i]
        k = normalize_kic(k)
        nu = safe_float(row.get('nu_max'))
        if nu is None or nu <= 0:
            warnings.append(f'{k}: missing nu_max')
            continue
        freq, power = raw_psd(row['psd_path'])
        axis, reconstructed = load_from_manifest_row(row, x.shape[-1], 'psd')
        original_input = x.detach().cpu().numpy().reshape(-1)
        if not np.allclose(reconstructed, original_input, rtol=1e-4, atol=2e-4):
            raise ValueError(f'{k}: preprocessing mismatch; aborting rather than using wrong physical replacement')
        # IMPORTANT: same mean/std as ORIGINAL PSD, even after modifying its power.
        logpower = np.log10(power + EPS)
        mu, sd = logpower.mean(), logpower.std()
        hump = (axis >= 0.5 * nu) & (axis <= 1.5 * nu)
        low = (axis >= 0.5 * nu / 3.0) & (axis < 0.5 * nu)
        if not hump.any() or not low.any():
            warnings.append(f'{k}: missing hump or low band')
            continue
        modifications = {}
        for name, mask in [('hump_zero', hump), ('low_equal_zero', low)]:
            v = original_input.copy()
            v[mask] = 0
            modifications[name] = (v, mask)
        duration = str(row.get('duration', config['data'].get('filter_duration', ''))).strip().lower()
        fit = fit_lookup.get((k, duration))
        if fit is None and args.fit_csv:
            warnings.append(f'{k}/{duration}: no matched two_fixed fit; skipping physical interventions')
        if fit is not None:
            try:
                bg, env = fit_parts(freq, fit)
                # Replace hump with fitted smooth Harvey+white background.
                bg_interp = transform_power(bg, freq, axis, mu, sd)
                v = original_input.copy()
                v[hump] = bg_interp[hump]
                modifications['hump_background'] = (v, hump)
                # Validate the physical fit on logarithmically binned PSD data.
                # The V3 optimizer fits median power in log-frequency bins.
                edges = np.geomspace(freq.min(), freq.max(), 181)
                bin_id = np.clip(np.searchsorted(edges, freq, side='right') - 1, 0, 179)
                bin_f, bin_p = [], []
                for ib in range(180):
                    inside = bin_id == ib
                    if inside.sum() >= 4:
                        bin_f.append(np.exp(np.log(freq[inside]).mean()))
                        bin_p.append(np.median(power[inside]))
                bin_f, bin_p = np.asarray(bin_f), np.asarray(bin_p)
                bg_b, env_b = fit_parts(bin_f, fit)
                total_b = bg_b + env_b
                outside = (bin_f < 0.5 * nu) | (bin_f > 1.5 * nu)
                residual = np.log10(np.maximum(bin_p, EPS)) - np.log10(np.maximum(total_b, EPS))
                outside_median = float(np.median(residual[outside])) if outside.any() else np.nan
                outside_rms = float(np.sqrt(np.mean(residual[outside]**2))) if outside.any() else np.nan
                diagnostics.append(dict(kic=k, duration=duration, model='two_fixed',
                    n_bins=int(len(bin_f)), n_bins_outside=int(outside.sum()),
                    outside_median_log10_data_over_model=outside_median,
                    outside_rms_log10_data_over_model=outside_rms,
                    fit_rms_log10_residual=float(fit.get('rms_log10_residual', np.nan)),
                    fit_file=str(fit.get('file', ''))))
                if not np.isfinite(outside_rms) or outside_rms > 0.5:
                    warnings.append(f'{k}/{duration}: poor total fit outside hump (RMS={outside_rms:.3f} dex); background intervention needs review')
                if count < args.plots:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.loglog(freq, power, '.', ms=1, alpha=.12, color='gray', label='Raw physical PSD')
                    ax.loglog(bin_f, bin_p, '.', ms=4, color='black', label='Log-bin medians')
                    ax.loglog(bin_f, bg_b, lw=1.5, label='Harvey + white background')
                    ax.loglog(bin_f, env_b, lw=1.2, label='Gaussian envelope')
                    ax.loglog(bin_f, total_b, lw=1.7, label='Total fitted model')
                    ax.axvspan(0.5*nu, 1.5*nu, alpha=.12, color='orange', label='Hump intervention')
                    ax.set(xlabel='Frequency (µHz)', ylabel='Physical PSD power',
                           title=f'KIC {k} ({duration}) — two_fixed fit; outside RMS {outside_rms:.3f} dex')
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(outdir / 'plots' / f'kic_{k}_{duration}_physical_fit.png', dpi=150)
                    plt.close(fig)
            except (ValueError, KeyError, FloatingPointError) as e:
                warnings.append(f'{k}: invalid physical fit: {e}')
        with torch.no_grad():
            inputs = np.stack([original_input] + [v for v, _ in modifications.values()])
            pred = model(torch.tensor(inputs, device=device, dtype=torch.float32).unsqueeze(1)).detach().cpu().numpy()
        if pred.ndim == 1:
            pred = pred[:, None]
        if pred.shape[1] != len(targets):
            raise ValueError('Unexpected model output shape')
        truth = np.asarray(y.detach().cpu().numpy()).reshape(-1)
        if len(truth) != len(targets):
            raise ValueError('Unexpected target shape')
        for j, target in enumerate(targets):
            base = float(pred[0, j])
            label = float(truth[j])
            is_class = target == 'class'
            for n, (name, (v, mask)) in enumerate(modifications.items(), start=1):
                changed = float(pred[n, j])
                probability = lambda z: float(1 / (1 + np.exp(-np.clip(z, -80, 80))))
                records.append(dict(kic=k, duration=duration, target=target, intervention=name,
                    nu_max_catalog_uHz=nu, n_bins_masked=int(mask.sum()), fraction_bins_masked=float(mask.mean()),
                    original_output=base, occluded_output=changed, delta_output=changed-base,
                    abs_delta_output=abs(changed-base), truth_model_units=label,
                    original_abs_error=abs(base-label) if not is_class else np.nan,
                    occluded_abs_error=abs(changed-label) if not is_class else np.nan,
                    delta_abs_error=(abs(changed-label)-abs(base-label)) if not is_class else np.nan,
                    original_probability=probability(base) if is_class else np.nan,
                    occluded_probability=probability(changed) if is_class else np.nan,
                    abs_delta_probability=abs(probability(changed)-probability(base)) if is_class else np.nan,
                    original_correct=(int((base >= 0) == (label >= 0.5))) if is_class else np.nan,
                    occluded_correct=(int((changed >= 0) == (label >= 0.5))) if is_class else np.nan,
                    prediction_flipped=int((base >= 0) != (changed >= 0)) if is_class else np.nan))
        if count < args.plots:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(axis, original_input, lw=.5, alpha=.65, label='Original standardized log PSD')
            for name, (v, _) in modifications.items():
                ax.plot(axis, v, lw=.8, alpha=.8, label=name)
            ax.set_xscale('log')
            ax.set(xlabel='Frequency (µHz)', ylabel='Model input', title=f'KIC {k} ({duration})')
            ax.axvline(nu, color='black', ls=':', lw=1)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / 'plots' / f'kic_{k}_{duration}.png', dpi=130)
            plt.close(fig)
        print(f'{count+1}/{len(indices)} KIC {k}: {list(modifications)}', flush=True)
    if len(diagnostics) != len(indices):
        warnings.append(f'Physical fit diagnostics available for {len(diagnostics)}/{len(indices)} selected stars')
    if not diagnostics:
        raise RuntimeError('No matched, valid two_fixed fits; check fit CSV, KIC and duration')
    if not records:
        raise RuntimeError('No valid interventions')
    pd.DataFrame(diagnostics).to_csv(outdir / 'fit_diagnostics.csv', index=False)
    result = pd.DataFrame(records)
    reference = result.drop_duplicates(['kic', 'target']).groupby('target')['original_output'].std(ddof=0)
    result['reference_output_std'] = result['target'].map(reference)
    result['abs_delta_in_output_std'] = result['abs_delta_output'] / result['reference_output_std'].replace(0, np.nan)
    result.to_csv(outdir / 'controlled_occlusion_per_star.csv', index=False)
    summary = summarize(result)
    summary.to_csv(outdir / 'controlled_occlusion_summary.csv', index=False)
    # Explicit accuracy and MAE comparisons in native training target units.
    metrics = []
    for (target, intervention), g in result.groupby(['target', 'intervention']):
        if target == 'class':
            metrics.append(dict(target=target, intervention=intervention, n=len(g),
                original_accuracy=float(g.original_correct.mean()),
                modified_accuracy=float(g.occluded_correct.mean()),
                delta_accuracy=float((g.occluded_correct-g.original_correct).mean())))
        else:
            metrics.append(dict(target=target, intervention=intervention, n=len(g),
                original_mae=float(g.original_abs_error.mean()),
                modified_mae=float(g.occluded_abs_error.mean()),
                delta_mae=float(g.delta_abs_error.mean())))
    pd.DataFrame(metrics).to_csv(outdir / 'controlled_occlusion_performance.csv', index=False)
    (outdir / 'settings.json').write_text(json.dumps(dict(run_dir=str(run_dir), checkpoint=str(checkpoint),
        fit_csv=str(args.fit_csv) if args.fit_csv else None,
        selected_kics=[normalize_kic(df.iloc[i]['kic']) for i in indices],
        equal_low_band='[nu_max/6, nu_max/2)', hump_band='[nu_max/2, 3*nu_max/2]',
        physical_intervention='Replace hump with two_fixed Harvey+white model using original PSD normalization; no Gaussian subtraction',
        note='Catalogue nu_max defines masks; outputs/errors are in model training target units; not independent test evaluation.'), indent=2))
    (outdir / 'warnings.txt').write_text('\n'.join(warnings) + '\n')
    print('\n', summary.to_string(index=False))
    print(f'\nSaved to {outdir}; warnings: {len(warnings)}')

if __name__ == '__main__':
    main()
