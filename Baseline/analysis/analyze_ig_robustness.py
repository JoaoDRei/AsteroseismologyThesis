#!/usr/bin/env python3
"""Compare existing IG maps with noise-averaged IG maps.

Run:
 python -m Baseline.analysis.analyze_ig_robustness

The defaults compare the I4 PSD ResNet1D multitask run and write results to
Baseline/results_analysis/ig_robustness. Each directory can be overridden with
the corresponding command-line option.

Only matching sample_id + target pairs are compared. Files are never modified.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Baseline.paths import BASELINE_DIR


DEFAULT_RUN_DIR = (
    BASELINE_DIR
    / 'results'
    / 'resnet1d_psd_80d_nu_max_delta_nu_class_20261018_190922_635749'
)
DEFAULT_IG_DIR = DEFAULT_RUN_DIR / 'interpretability'
DEFAULT_SMOOTHGRAD_IG_DIR = DEFAULT_RUN_DIR / 'interpretability_robustness'
DEFAULT_OUTPUT_DIR = BASELINE_DIR / 'results_analysis' / 'ig_robustness'


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            'attr': np.asarray(data['attribution'], dtype=np.float64).ravel(),
            'axis': np.asarray(data['input_axis'], dtype=np.float64).ravel(),
            'input': np.asarray(data['model_input']),
            'baseline': np.asarray(data['baseline']),
            'metadata': json.loads(str(data['metadata'].item())),
        }


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def top_overlap(a, b, fraction=0.1):
    n = max(1, math.ceil(len(a) * fraction))
    ia = np.argpartition(np.abs(a), -n)[-n:]
    ib = np.argpartition(np.abs(b), -n)[-n:]
    return float(np.intersect1d(ia, ib).size / n)


def localisation(a, axis, nu_max):
    result = {}
    weights = np.abs(a)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return {f'fraction_{band}': np.nan for band in ('wide', 'medium', 'narrow')}
    for band, lo, hi in [('wide', 0.5, 1.5), ('medium', 0.75, 1.25), ('narrow', 0.9, 1.1)]:
        mask = (axis >= lo * nu_max) & (axis <= hi * nu_max)
        result[f'fraction_{band}'] = float(weights[mask].sum() / total)
    return result


def compare(ig_path, smooth_path, plots_dir, make_plots=True):
    original, averaged = load(ig_path), load(smooth_path)
    m1, m2 = original['metadata'], averaged['metadata']
    if m1.get('method') != 'ig' or m2.get('method') != 'smoothgrad_ig':
        raise ValueError('Unexpected attribution method metadata')
    for key in ('sample_id', 'kic', 'duration', 'target', 'target_index', 'model_type', 'input_mode', 'checkpoint_path', 'baseline_type'):
        if m1.get(key) != m2.get(key):
            raise ValueError(f'{key} mismatch: {ig_path.name} / {smooth_path.name}')
    if m1.get('n_steps') != m2.get('n_steps'):
        raise ValueError(f'IG integration step mismatch for {ig_path.name}')
    for key in ('input', 'baseline', 'axis'):
        if not np.array_equal(original[key], averaged[key]):
            raise ValueError(f'{key} arrays differ for {ig_path.name}')
    a, b = original['attr'], averaged['attr']
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError(f'Invalid attribution arrays for {ig_path.name}')
    if not len(a):
        raise ValueError(f'Empty attribution for {ig_path.name}')
    pearson = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    spearman = float(pd.Series(np.abs(a)).corr(pd.Series(np.abs(b)), method='spearman'))
    result = {
        'sample_id': m1.get('sample_id'), 'kic': str(m1['kic']),
        'duration': m1.get('duration'), 'target': m1['target'],
        'model_type': m1['model_type'], 'input_mode': m1['input_mode'],
        'n_bins': len(a), 'ig_n_steps': m1.get('n_steps'),
        'noise_samples': m2.get('n_samples'), 'noise_std': m2.get('noise_std'),
        'pearson_signed': pearson,
        'spearman_absolute': spearman,
        'cosine_signed': cosine(a, b),
        'cosine_absolute': cosine(np.abs(a), np.abs(b)),
        'top_10pct_overlap': top_overlap(a, b),
        'ig_file': str(ig_path), 'smoothgrad_ig_file': str(smooth_path),
    }
    nu = m1.get('nu_max')
    try:
        nu = float(nu)
    except (TypeError, ValueError):
        nu = np.nan
    if m1.get('input_mode') == 'psd' and np.isfinite(nu) and nu > 0:
        loc_a = localisation(a, original['axis'], nu)
        loc_b = localisation(b, original['axis'], nu)
        for key in loc_a:
            result['ig_' + key] = loc_a[key]
            result['smoothgrad_ig_' + key] = loc_b[key]
            result['delta_' + key] = loc_b[key] - loc_a[key]
    if make_plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        axis = original['axis']
        aa = np.abs(a) / max(np.max(np.abs(a)), np.finfo(float).tiny)
        bb = np.abs(b) / max(np.max(np.abs(b)), np.finfo(float).tiny)
        ax.plot(axis, aa, label='IG |attribution|', alpha=0.7, lw=0.8)
        ax.plot(axis, bb, label='Noise-averaged IG |attribution|', alpha=0.7, lw=0.8)
        if m1.get('input_mode') == 'psd':
            ax.set_xscale('log')
            ax.set_xlabel('Frequency (µHz)')
            if np.isfinite(nu) and nu > 0:
                ax.axvspan(0.5 * nu, 1.5 * nu, alpha=0.10, color='green', label='0.5–1.5 νmax')
        else:
            ax.set_xlabel('Time (days)')
        ax.set_ylabel('Absolute attribution / own maximum')
        ax.set_title(f"{m1.get('sample_id')} | {m1['target']} | Pearson={pearson:.3f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{ig_path.stem.removesuffix('_ig')}_overlay.png", dpi=130)
        plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ig-dir', type=Path, default=DEFAULT_IG_DIR)
    parser.add_argument(
        '--smoothgrad-ig-dir',
        type=Path,
        default=DEFAULT_SMOOTHGRAD_IG_DIR,
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    ig_dir = args.ig_dir.expanduser().resolve()
    sg_dir = args.smoothgrad_ig_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not ig_dir.is_dir() or not sg_dir.is_dir():
        parser.error('Both attribution directories must exist')
    out.mkdir(parents=True, exist_ok=True)
    plots = out / 'plots'
    if not args.no_plots:
        plots.mkdir(exist_ok=True)
    ig_paths = {p.stem[:-3]: p for p in ig_dir.rglob('kic_*_ig.npz')}
    sg_paths = {p.stem[:-len('_smoothgrad_ig')]: p for p in sg_dir.rglob('kic_*_smoothgrad_ig.npz')}
    common = sorted(ig_paths.keys() & sg_paths.keys())
    print(f'IG files: {len(ig_paths)} | Noise-averaged IG files: {len(sg_paths)} | Matched: {len(common)}')
    if not common:
        raise SystemExit('No matching files. Check directories and sample/target naming.')
    records = []
    for stem in common:
        records.append(compare(ig_paths[stem], sg_paths[stem], plots, not args.no_plots))
    per_star = pd.DataFrame(records)
    per_star.to_csv(out / 'ig_robustness_per_star.csv', index=False)
    groups = ['model_type', 'input_mode', 'target']
    metrics = ['pearson_signed', 'spearman_absolute', 'cosine_signed', 'cosine_absolute', 'top_10pct_overlap']
    metrics += [c for c in per_star.columns if c.startswith(('ig_fraction_', 'smoothgrad_ig_fraction_', 'delta_fraction_'))]
    summary = per_star.groupby(groups, dropna=False).agg(
        n_samples=('sample_id', 'count'), n_unique_kics=('kic', 'nunique'),
        **{f'median_{m}': (m, 'median') for m in metrics},
    ).reset_index()
    summary.to_csv(out / 'ig_robustness_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'\nSaved results to {out}')
    if len(common) != len(ig_paths) or len(common) != len(sg_paths):
        print('Note: unmatched attribution files were excluded.')


if __name__ == '__main__':
    main()
