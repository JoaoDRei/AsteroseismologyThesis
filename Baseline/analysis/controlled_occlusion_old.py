#!/usr/bin/env python3
"""Controlled PSD occlusion using the EXISTING trained model.

Run from project root:
 python -m Baseline.analysis.controlled_occlusion

By default this uses the PSD ResNet1D multitask run, the ten IG robustness
pilot stars, and the V3 two_free fitted parameters. Outputs are saved in the
run's controlled_occlusion/ directory and mirrored to
Baseline/results_analysis/controlled_occlusion/.

The fit CSV and all defaults can be overridden with command-line options.
Without --fit-csv, zero-mask controls still run. No retraining and no changes
to original inputs/checkpoints.
"""
from __future__ import annotations
import argparse
import json
import shutil
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
    BASELINE_DIR
    / 'results'
    / 'resnet1d_psd_80d_nu_max_delta_nu_class_20261018_190922_635749'
)
DEFAULT_FIT_CSV = (
    BASELINE_DIR.parent
    / 'data'
    / 'fitted_parameters_v3_two_free_all'
    / 'fit_results.csv'
)
DEFAULT_KICS = (
    '11245339,11805876,12784948,1429505,3533464,'
    '4348593,4991033,5172229,6501869,9021799'
)
DEFAULT_ANALYSIS_OUTPUT_DIR = (
    BASELINE_DIR / 'results_analysis' / 'controlled_occlusion'
)

def fit_parts(freq, row):
    """Reconstruct the two_free Harvey + white + Gaussian model in native PSD units."""
    if str(row['model']) != 'two_free':
        raise ValueError('Only two_free fit rows are supported')
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
    ap.add_argument(
        '--fit-csv',
        type=Path,
        default=DEFAULT_FIT_CSV,
        help='V3 two_free fit_results.csv; defaults to the granulation fit table',
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        help='Primary output directory; defaults to <run-dir>/controlled_occlusion',
    )
    ap.add_argument(
        '--kics',
        type=str,
        default=DEFAULT_KICS,
        help='Comma-separated KICs; defaults to the IG robustness pilot stars',
    )
    ap.add_argument('--max-stars', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--plots', type=int, default=3, help='Number of illustrative PSD plots')
    args = ap.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / 'config.yaml'
    with config_path.open() as f:
        config = yaml.safe_load(f)
    if config['data']['mode'] != 'psd':
        raise ValueError('PSD runs only')
    requested = (
        [value.strip() for value in args.kics.split(',') if value.strip()]
        if args.kics is not None
        else None
    )
    if args.kics is not None and not requested:
        ap.error('--kics cannot be empty')
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
    fit_csv = args.fit_csv.expanduser().resolve() if args.fit_csv else None
    if fit_csv:
        if not fit_csv.is_file():
            raise FileNotFoundError(f'Fit CSV not found: {fit_csv}')
        fits = pd.read_csv(fit_csv)
        fits = fits[(fits['model'] == 'two_free') & (fits['success'].astype(str).str.lower().isin(['true', '1']))]
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
    outdir = (
        args.output_dir or run_dir / 'controlled_occlusion'
    ).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'plots').mkdir(exist_ok=True)
    targets = list(config['model']['targets'])
    records, warnings = [], []
    width = np.log(3.0)
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
            warnings.append(f'{k}/{duration}: no matched two_free fit; skipping physical interventions')
        if fit is not None:
            try:
                bg, env = fit_parts(freq, fit)
                # Replace hump with fitted smooth Harvey+white background.
                bg_interp = transform_power(bg, freq, axis, mu, sd)
                v = original_input.copy()
                v[hump] = bg_interp[hump]
                modifications['hump_background'] = (v, hump)
                # Subtract only fitted Gaussian envelope from physical PSD.
                # Clip to a small positive floor; record clip fraction.
                floor = max(np.percentile(power[power > 0], 1) * 1e-3, EPS)
                sub = np.maximum(power - env, floor)
                clip_fraction = float(np.mean(power - env < floor))
                if clip_fraction > 0.10:
                    warnings.append(f'{k}: Gaussian subtraction clipped {clip_fraction:.1%} of raw bins; interpret cautiously')
                v = transform_power(sub, freq, axis, mu, sd)
                modifications['gaussian_subtraction'] = (v, np.abs(v-original_input) > 1e-7)
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
    if not records:
        raise RuntimeError('No valid interventions')
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
        fit_csv=str(fit_csv) if fit_csv else None,
        selected_kics=[normalize_kic(df.iloc[i]['kic']) for i in indices],
        equal_low_band='[nu_max/6, nu_max/2)', hump_band='[nu_max/2, 3*nu_max/2]',
        gaussian_subtraction='physical power minus fitted Gaussian, clipped positive, original normalization',
        note='Catalogue nu_max defines masks; outputs/errors are in model training target units; not independent test evaluation.'), indent=2))
    (outdir / 'warnings.txt').write_text('\n'.join(warnings) + '\n')
    analysis_outdir = DEFAULT_ANALYSIS_OUTPUT_DIR.resolve()
    if analysis_outdir != outdir:
        analysis_outdir.mkdir(parents=True, exist_ok=True)
        for path in outdir.iterdir():
            if path.is_file():
                shutil.copy2(path, analysis_outdir / path.name)
        analysis_plots = analysis_outdir / 'plots'
        analysis_plots.mkdir(exist_ok=True)
        for path in (outdir / 'plots').glob('*.png'):
            shutil.copy2(path, analysis_plots / path.name)
    print('\n', summary.to_string(index=False))
    print(f'\nSaved run results to {outdir}; warnings: {len(warnings)}')
    if analysis_outdir != outdir:
        print(f'Mirrored results to {analysis_outdir}')

if __name__ == '__main__':
    main()
