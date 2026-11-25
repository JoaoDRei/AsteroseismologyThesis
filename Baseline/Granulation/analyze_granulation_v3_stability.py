#!/usr/bin/env python3
"""V3 cross-duration stability and cross-model sensitivity diagnostics.

Run from anywhere:
    python analyze_granulation_v3_stability.py
    python analyze_granulation_v3_stability.py --input data/fitted_parameters_v3_pilot_300/fit_results.csv

Uses existing V3 fit_results.csv only; does NOT refit PSDs.
All ratios are calculated on matched KICs, never across different stars.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # consistent with fit_granulation_v3.py
DURATIONS = ('20d', '55d', '80d')
MODELS = ('two_fixed', 'two_free')
PARAMETERS = {
    'long_nu_char': 'long_component_nu_char_uHz',
    'short_nu_char': 'short_component_nu_char_uHz',
    'long_rms': 'long_integrated_rms_native_sqrt_uHz',
    'short_rms': 'short_integrated_rms_native_sqrt_uHz',
}
FLAGS = ('long_alpha_at_bound', 'short_alpha_at_bound',
         'long_char_below_fit_band', 'short_char_below_fit_band',
         'long_char_below_filter_timescale', 'short_char_below_filter_timescale')


def truthy(series):
    return series.astype('string').str.strip().str.lower().isin(['true', '1', 'yes'])


def numeric_positive(s):
    x = pd.to_numeric(s, errors='coerce')
    return x.where(np.isfinite(x) & (x > 0))


def summary_stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0, 'median_abs_log10_ratio': np.nan, 'median_percent_multiplicative_difference': np.nan,
                'fraction_over_20pct': np.nan, 'fraction_over_50pct': np.nan,
                'fraction_over_factor2': np.nan, 'p90_abs_log10_ratio': np.nan}
    a = np.abs(x)
    factor = 10 ** a
    return {'n': int(len(x)), 'median_abs_log10_ratio': float(np.median(a)),
            'median_percent_multiplicative_difference': float(100 * (10 ** np.median(a) - 1)),
            'fraction_over_20pct': float(np.mean(factor > 1.2)),
            'fraction_over_50pct': float(np.mean(factor > 1.5)),
            'fraction_over_factor2': float(np.mean(factor > 2)),
            'p90_abs_log10_ratio': float(np.percentile(a, 90))}


def add_rows(rows, group, model, pair, parameter, values):
    rows.append({'comparison': group, 'model': model, 'pair': pair, 'parameter': parameter,
                 **summary_stats(values)})


def plot_distribution(df, output, xlabel, title, column='abs_log10_ratio'):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby('model', sort=True):
        x = pd.to_numeric(group[column], errors='coerce').to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x):
            ax.hist(x, bins=np.linspace(0, min(max(np.percentile(x, 99), .35), 1.5), 45),
                    histtype='step', linewidth=1.8, density=True, label=f'{label} (n={len(x)})')
    ax.set(xlabel=xlabel, ylabel='Density', title=title)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=ROOT / 'data/fitted_parameters_v3_pilot_300/fit_results.csv')
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--max-plot-points', type=int, default=3000)
    args = ap.parse_args()
    if not args.input.exists():
        ap.error(f'Input file not found: {args.input}. Supply --input explicitly.')
    out = args.output_dir or args.input.parent / 'stability_analysis'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    required = ['kic', 'duration', 'model', 'success', 'nu_max_catalogue', *PARAMETERS.values()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        ap.error(f'Missing V3 columns: {missing}')
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df['duration'] = df.duration.astype(str).str.lower().str.strip()
    df['model'] = df.model.astype(str).str.strip()
    df = df[truthy(df.success) & df.model.isin(args.models) & df.duration.isin(DURATIONS) & df.kic.notna()].copy()
    if df.empty:
        ap.error('No successful two-component V3 fits found.')
    df['kic'] = df.kic.astype('int64')
    for col in PARAMETERS.values():
        df[col] = numeric_positive(df[col])
    df['nu_max_catalogue'] = numeric_positive(df['nu_max_catalogue'])
    for flag in FLAGS:
        df[flag] = truthy(df[flag]) if flag in df else False
    dup = df.duplicated(['kic', 'duration', 'model'], keep=False)
    if dup.any():
        sample = df.loc[dup, ['kic', 'duration', 'model']].head(8).to_dict('records')
        ap.error(f'Duplicate KIC/duration/model rows (must resolve rather than silently deduplicate): {sample}')

    # Each row is one matched pair of duration fits for the SAME star and model.
    pair_rows = []
    summary_rows = []
    for model in args.models:
        subset = df[df.model == model].set_index(['kic', 'duration'])
        for left, right in [('20d', '55d'), ('55d', '80d'), ('20d', '80d')]:
            a = subset.xs(left, level='duration', drop_level=True) if left in subset.index.get_level_values('duration') else pd.DataFrame()
            b = subset.xs(right, level='duration', drop_level=True) if right in subset.index.get_level_values('duration') else pd.DataFrame()
            if a.empty or b.empty:
                continue
            shared = a.index.intersection(b.index)
            for kic in shared:
                for label, col in PARAMETERS.items():
                    x, y = a.at[kic, col], b.at[kic, col]
                    log_ratio = float(np.log10(x / y)) if np.isfinite(x) and np.isfinite(y) else np.nan
                    pair_rows.append({'kic': int(kic), 'model': model, 'pair': f'{left}_vs_{right}',
                                      'parameter': label, 'left_value': x, 'right_value': y,
                                      'log10_ratio': log_ratio, 'abs_log10_ratio': abs(log_ratio),
                                      'nu_max_catalogue': a.at[kic, 'nu_max_catalogue'],
                                      'either_long_below_band': bool(a.at[kic, 'long_char_below_fit_band'] or b.at[kic, 'long_char_below_fit_band']),
                                      'either_long_alpha_bound': bool(a.at[kic, 'long_alpha_at_bound'] or b.at[kic, 'long_alpha_at_bound'])})
            for label in PARAMETERS:
                values = [r['log10_ratio'] for r in pair_rows if r['model'] == model and r['pair'] == f'{left}_vs_{right}' and r['parameter'] == label]
                add_rows(summary_rows, 'cross_duration', model, f'{left}_vs_{right}', label, values)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / 'cross_duration_pairs.csv', index=False)

    # Cross-model sensitivity, paired by identical KIC and duration.
    between_rows = []
    if set(MODELS).issubset(args.models):
        fixed = df[df.model == 'two_fixed'].set_index(['kic', 'duration'])
        free = df[df.model == 'two_free'].set_index(['kic', 'duration'])
        for kic, duration in fixed.index.intersection(free.index):
            for label, col in PARAMETERS.items():
                x, y = fixed.at[(kic, duration), col], free.at[(kic, duration), col]
                lr = float(np.log10(x / y)) if np.isfinite(x) and np.isfinite(y) else np.nan
                between_rows.append({'kic': int(kic), 'duration': duration, 'model': 'fixed_vs_free',
                                     'parameter': label, 'fixed_value': x, 'free_value': y,
                                     'log10_ratio': lr, 'abs_log10_ratio': abs(lr),
                                     'nu_max_catalogue': fixed.at[(kic, duration), 'nu_max_catalogue']})
        between = pd.DataFrame(between_rows)
        for (duration, label), g in between.groupby(['duration', 'parameter']):
            add_rows(summary_rows, 'cross_model', 'fixed_vs_free', duration, label, g.log10_ratio)
    else:
        between = pd.DataFrame(columns=['kic', 'duration', 'model', 'parameter', 'log10_ratio', 'abs_log10_ratio'])
    between.to_csv(out / 'cross_model_pairs.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'stability_summary.csv', index=False)

    # Flag prevalence; fixed-alpha alpha-bound flags are not meaningful and shown as N/A.
    flag_rows = []
    for (model, duration), g in df.groupby(['model', 'duration']):
        row = {'model': model, 'duration': duration, 'n': len(g)}
        for flag in FLAGS:
            row[flag + '_fraction'] = (np.nan if model == 'two_fixed' and 'alpha_at_bound' in flag
                                      else float(g[flag].mean()))
        flag_rows.append(row)
    pd.DataFrame(flag_rows).to_csv(out / 'flag_prevalence.csv', index=False)

    # Flag unstable stars; use all three durations when available and keep incomplete ones explicit.
    star_rows = []
    for (model, kic), g in df.groupby(['model', 'kic']):
        row = {'model': model, 'kic': int(kic), 'n_durations': len(g),
               'durations': ','.join(sorted(g.duration)), 'nu_max_catalogue': float(g.nu_max_catalogue.median()),
               'any_long_below_band': bool(g.long_char_below_fit_band.any()),
               'any_long_below_filter_timescale': bool(g.long_char_below_filter_timescale.any()),
               'any_long_alpha_bound': bool(g.long_alpha_at_bound.any()) if model == 'two_free' else False}
        for label, col in PARAMETERS.items():
            vals = g[col].dropna().to_numpy(float)
            row[label + '_max_over_min'] = float(vals.max() / vals.min()) if len(vals) >= 2 else np.nan
            row[label + '_range_dex'] = float(np.log10(vals.max() / vals.min())) if len(vals) >= 2 else np.nan
        row['short_unstable_factor2'] = bool(row['short_nu_char_max_over_min'] > 2 or row['short_rms_max_over_min'] > 2) if len(g) >= 2 else False
        row['long_unstable_factor2'] = bool(row['long_nu_char_max_over_min'] > 2 or row['long_rms_max_over_min'] > 2) if len(g) >= 2 else False
        star_rows.append(row)
    stars = pd.DataFrame(star_rows)
    stars.to_csv(out / 'per_star_stability.csv', index=False)

    # Frequency-dependence diagnostics; descriptive plots only, no unjustified p-values.
    for label in PARAMETERS:
        q = pairs[(pairs.parameter == label) & (pairs.pair == '20d_vs_80d')]
        plot_distribution(q, out / f'cross_duration_{label}.png',
                          '|log10(20d / 80d)|', f'Cross-duration stability: {label}')
        if not q.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model, g in q.groupby('model'):
                x = pd.to_numeric(g.nu_max_catalogue, errors='coerce').to_numpy(float)
                y = pd.to_numeric(g.abs_log10_ratio, errors='coerce').to_numpy(float)
                valid = np.isfinite(x) & (x > 0) & np.isfinite(y)
                ax.scatter(x[valid][:args.max_plot_points], y[valid][:args.max_plot_points],
                           s=13, alpha=.45, label=model)
            ax.set_xscale('log')
            ax.set(xlabel=r'Catalogue $\nu_{\max}$ (µHz)', ylabel='|log10(20d / 80d)|',
                   title=rf'Stability versus $\nu_{{\max}}$: {label}')
            ax.legend()
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(out / f'stability_vs_numax_{label}.png', dpi=150)
            plt.close(fig)
        if not between.empty:
            q = between[(between.parameter == label) & (between.duration == '55d')]
            plot_distribution(q, out / f'cross_model_{label}_55d.png',
                              '|log10(two_fixed / two_free)|', f'Model sensitivity (55d): {label}')

    # A compact plain-text report, explicit denominators and incomplete coverage.
    lines = [f'V3 stability diagnostics', f'Input: {args.input.resolve()}',
             f'Successful fit rows: {len(df)}', f'Unique KICs: {df.kic.nunique()}',
             'NOTE: durations share underlying light curves; not independent replicates.',
             'NOTE: integrated RMS is a model-dependent extrapolated amplitude proxy in native units.',
             'NOTE: 20d/80d stability does not establish absolute physical correctness.', '',
             'CROSS-DURATION COMPARISONS (matched KICs):']
    for _, r in summary[summary.comparison == 'cross_duration'].iterrows():
        lines.append(f"{r['model']:10s} {r['pair']:12s} {r['parameter']:14s} "
                     f"n={int(r['n']):3d} median |dex|={r['median_abs_log10_ratio']:.4f} "
                     f">20%={r['fraction_over_20pct']:.1%} >50%={r['fraction_over_50pct']:.1%} "
                     f">2x={r['fraction_over_factor2']:.1%}")
    lines += ['', 'CROSS-MODEL COMPARISONS (matched KIC and duration):']
    for _, r in summary[summary.comparison == 'cross_model'].iterrows():
        lines.append(f"{r['pair']:3s} {r['parameter']:14s} n={int(r['n']):3d} "
                     f"median |dex|={r['median_abs_log10_ratio']:.4f} "
                     f">20%={r['fraction_over_20pct']:.1%} >2x={r['fraction_over_factor2']:.1%}")
    lines += ['', 'PER-STAR FACTOR-2 INSTABILITY (>=2 durations):']
    for model, g in stars[stars.n_durations >= 2].groupby('model'):
        lines.append(f'{model}: n={len(g)} short={g.short_unstable_factor2.mean():.1%} '
                     f'long={g.long_unstable_factor2.mean():.1%}')
    lines += ['', 'SELECTED EXAMPLES (if present):']
    for kic in [1865747, 2305145, 2423737, 2854994, 3222670]:
        for _, r in stars[stars.kic == kic].iterrows():
            lines.append(f"KIC {kic} {r['model']}: short nu range factor={r['short_nu_char_max_over_min']:.2f}, "
                         f"short RMS factor={r['short_rms_max_over_min']:.2f}, "
                         f"long nu factor={r['long_nu_char_max_over_min']:.2f}")
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'run_config.json').write_text(json.dumps({'input': str(args.input.resolve()),
         'models': args.models, 'durations': DURATIONS, 'parameters': PARAMETERS,
         'thresholds': {'20pct': 1.2, '50pct': 1.5, 'factor2': 2}}, indent=2))
    print('\n'.join(lines[:7]))
    print(f'\nSaved diagnostics to: {out.resolve()}')

if __name__ == '__main__':
    main()
