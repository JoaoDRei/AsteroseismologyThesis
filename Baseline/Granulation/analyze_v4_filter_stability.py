#!/usr/bin/env python3
"""Compare cross-filter stability of the full V3 two- and V4 three-component fits.
Run from project root: python Baseline/Granulation/analyze_v4_filter_stability.py
Only reads the existing fit_results.csv files; does not refit any PSDs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------- EDIT THESE PATHS IF NEEDED -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIT_CSVS = {
    'two_fixed': PROJECT_ROOT / 'data/fitted_parameters_v3_two_fixed_all/fit_results.csv',
    'three_fixed': PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_all/fit_results.csv',
}
OUTPUT_DIR = PROJECT_ROOT / 'results/granulation_v4_filter_stability_full'
FILTERS = ('20d', '55d', '80d')
# These are deliberately broad, exploratory plausibility windows, NOT
# identification criteria established by Corsaro et al.
MID_NU_MAX_RANGE = (0.15, 0.60)
HIGH_NU_MAX_RANGE = (0.50, 1.80)
MID_HIGH_RATIO_RANGE = (0.20, 0.50)
# Stable means max/min across the three filters <= 1 + tolerance.
STABILITY_TOLERANCES = (0.05, 0.10, 0.20)
MIN_RMS = 1e-12
# -------------------------------------------------------------------------

COMPONENTS = {
    'two_fixed': ('long', 'short'),
    'three_fixed': ('low', 'mid', 'high'),
}

def number(s):
    return pd.to_numeric(s, errors='coerce')

def flag(s):
    return s.astype('string').str.strip().str.lower().isin(('true', '1', 'yes'))

def median_or_none(s):
    v = pd.to_numeric(s, errors='coerce').dropna()
    return float(v.median()) if len(v) else None

def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def three_filter_table(df, model, comp, param, suffix):
    subset = df.loc[df.model.eq(model), ['kic', 'duration', param]].copy()
    subset[param] = number(subset[param])
    p = subset.pivot(index='kic', columns='duration', values=param).reindex(columns=list(FILTERS))
    p.columns = [f'{model}_{comp}_{suffix}_{c}' for c in FILTERS]
    return p

def summarize_metric(p, prefix):
    vals = p.to_numpy(dtype=float)
    valid = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
    mx = np.full(len(p), np.nan)
    cv = np.full(len(p), np.nan)
    if valid.any():
        a = vals[valid]
        mx[valid] = a.max(axis=1) / a.min(axis=1) - 1.0
        cv[valid] = a.std(axis=1, ddof=0) / a.mean(axis=1)
    return pd.DataFrame({f'{prefix}_range_fraction': mx,
                         f'{prefix}_cv': cv,
                         f'{prefix}_complete': valid}, index=p.index)

def main():
    for model, path in FIT_CSVS.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {model} results: {path}')
    v4_dir = FIT_CSVS['three_fixed'].parent
    completion_markers = (
        v4_dir / 'model_comparison.csv',
        v4_dir / 'paired_holdout_rms.csv',
        v4_dir / 'paired_summary.json',
    )
    missing_markers = [path.name for path in completion_markers if not path.is_file()]
    if missing_markers:
        raise RuntimeError(
            'The V4 fit appears incomplete; wait for fit_granulation_v4_full.py '
            'to finish before running this analysis. Missing final output(s): '
            + ', '.join(missing_markers)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for model, comps in COMPONENTS.items():
        source = pd.read_csv(FIT_CSVS[model], low_memory=False)
        required = {'kic', 'duration', 'model', 'success', 'nu_max_catalogue',
                    'holdout_rms_log10_residual'}
        for comp in comps:
            required.update((f'{comp}_component_nu_char_uHz',
                             f'{comp}_integrated_rms_native_sqrt_uHz'))
        missing = required - set(source.columns)
        if missing:
            raise ValueError(
                f'{FIT_CSVS[model]} is missing columns: '
                + ', '.join(sorted(missing))
            )
        source = source.loc[source.model.eq(model)].copy()
        if source.empty:
            raise ValueError(f'No {model} rows found in {FIT_CSVS[model]}')
        frames.append(source)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df['kic'] = number(df.kic).astype('Int64')
    df = df[df.kic.notna() & df.duration.isin(FILTERS) & df.model.isin(COMPONENTS)].copy()
    df['success'] = flag(df.success)
    df = df[df.success].copy()
    if df.duplicated(['kic', 'duration', 'model']).any():
        raise ValueError('Duplicate KIC/duration/model combinations. Resolve before analysis.')
    df['nu_max_catalogue'] = number(df.nu_max_catalogue)
    df['holdout_rms_log10_residual'] = number(df.holdout_rms_log10_residual)
    stars = pd.DataFrame(index=pd.Index(sorted(df.kic.unique()), name='kic'))
    stars['nu_max_catalogue'] = df.groupby('kic').nu_max_catalogue.median()
    stats = {}
    for model, comps in COMPONENTS.items():
        for comp in comps:
            for suffix, col in (('freq', f'{comp}_component_nu_char_uHz'),
                                ('rms', f'{comp}_integrated_rms_native_sqrt_uHz')):
                p = three_filter_table(df, model, comp, col, suffix)
                stars = stars.join(p)
                cols = [f'{model}_{comp}_{suffix}_{c}' for c in FILTERS]
                stars = stars.join(summarize_metric(stars[cols], f'{model}_{comp}_{suffix}'))
                key = f'{model}_{comp}_{suffix}'
                complete = stars[f'{key}_complete']
                s = stars.loc[complete, f'{key}_range_fraction'].dropna()
                stats[key] = {
                    'n_complete_stars': int(len(s)),
                    'median_max_over_min_minus_1': median_or_none(s),
                    'p90_max_over_min_minus_1': float(s.quantile(.9)) if len(s) else None,
                    'median_cv': median_or_none(stars.loc[complete, f'{key}_cv']),
                    **{f'fraction_within_{int(t*100)}pct': float((s <= t).mean()) if len(s) else None
                       for t in STABILITY_TOLERANCES},
                }
        hold = df[df.model.eq(model)].pivot(index='kic', columns='duration',
                  values='holdout_rms_log10_residual').reindex(columns=list(FILTERS))
        hold.columns = [f'{model}_holdout_{c}' for c in FILTERS]
        stars = stars.join(hold)

    # Each row corresponds to a distinct star, not 3 correlated PSDs.
    for model, comps in COMPONENTS.items():
        for comp in comps:
            for filt in FILTERS:
                col = f'{model}_{comp}_freq_{filt}'
                stars[f'{model}_{comp}_nu_over_numax_{filt}'] = stars[col] / stars.nu_max_catalogue

    for filt in FILTERS:
        mid = stars[f'three_fixed_mid_nu_over_numax_{filt}']
        high = stars[f'three_fixed_high_nu_over_numax_{filt}']
        ratio = (stars[f'three_fixed_mid_freq_{filt}'] /
                 stars[f'three_fixed_high_freq_{filt}'])
        stars[f'mid_high_freq_ratio_{filt}'] = ratio
        stars[f'plausible_{filt}'] = (mid.between(*MID_NU_MAX_RANGE) &
              high.between(*HIGH_NU_MAX_RANGE) & ratio.between(*MID_HIGH_RATIO_RANGE))
        # Below-band/filter flags are informational; never infer identifiability
        # of the mid component solely from the low-component flag.
        src = df[df.model.eq('three_fixed')].set_index(['kic', 'duration'])
        for field in ('low_char_below_fit_band', 'low_char_below_filter_timescale',
                      'mid_char_below_fit_band', 'mid_char_below_filter_timescale'):
            if field in df.columns:
                stars[f'{field}_{filt}'] = [bool(flag(pd.Series([src.at[(k, filt), field]])).iloc[0])
                    if (k, filt) in src.index else False for k in stars.index]

    stars['plausible_all_three'] = stars[[f'plausible_{x}' for x in FILTERS]].all(axis=1)
    stars['mid_frequency_stable_10pct'] = stars['three_fixed_mid_freq_range_fraction'].le(.10)
    stars['mid_amplitude_stable_10pct'] = stars['three_fixed_mid_rms_range_fraction'].le(.10)
    stars['mid_stable_both_10pct'] = (stars.mid_frequency_stable_10pct &
                                      stars.mid_amplitude_stable_10pct)
    stars['mid_stable_and_plausible'] = stars.mid_stable_both_10pct & stars.plausible_all_three
    stars['low_below_band_any'] = stars[[f'low_char_below_fit_band_{x}' for x in FILTERS]
                                         ].any(axis=1) if 'low_char_below_fit_band_20d' in stars else False

    # Paired comparison of two_fixed long versus three_fixed mid, and
    # two_fixed short versus three_fixed high, on the same complete stars.
    paired = {}
    for old, new in (('long', 'mid'), ('short', 'high')):
        for kind in ('freq', 'rms'):
            a = stars[f'two_fixed_{old}_{kind}_range_fraction']
            b = stars[f'three_fixed_{new}_{kind}_range_fraction']
            mask = a.notna() & b.notna()
            paired[f'two_{old}_vs_three_{new}_{kind}'] = {
                'n_paired_stars': int(mask.sum()),
                'median_two_range_fraction': median_or_none(a[mask]),
                'median_three_range_fraction': median_or_none(b[mask]),
                'fraction_three_more_stable': float((b[mask] < a[mask]).mean()) if mask.any() else None,
            }

    # Matched 20d/55d/80d predictive fit comparison.
    hold = {}
    for filt in FILTERS:
        a = stars[f'two_fixed_holdout_{filt}']
        b = stars[f'three_fixed_holdout_{filt}']
        mask = a.notna() & b.notna()
        hold[filt] = {'n': int(mask.sum()),
                      'fraction_three_better': float((b[mask] < a[mask]).mean()) if mask.any() else None,
                      'median_paired_improvement_dex': median_or_none(a[mask] - b[mask])}

    complete_mid = (stars.three_fixed_mid_freq_complete & stars.three_fixed_mid_rms_complete)
    subset = stars[complete_mid]
    two_fixed_kics = set(df.loc[df.model.eq('two_fixed'), 'kic'])
    three_fixed_kics = set(df.loc[df.model.eq('three_fixed'), 'kic'])
    overall = {
        'n_unique_stars': len(stars),
        'n_two_fixed_stars': len(two_fixed_kics),
        'n_three_fixed_stars': len(three_fixed_kics),
        'n_matched_stars': len(two_fixed_kics & three_fixed_kics),
        'n_complete_three_fixed_mid': int(len(subset)),
        'n_mid_stable_both_10pct': int(subset.mid_stable_both_10pct.sum()),
        'fraction_mid_stable_both_10pct': float(subset.mid_stable_both_10pct.mean()) if len(subset) else None,
        'n_mid_stable_and_plausible': int(subset.mid_stable_and_plausible.sum()),
        'fraction_mid_stable_and_plausible': float(subset.mid_stable_and_plausible.mean()) if len(subset) else None,
        'fraction_low_below_band_any': float(subset.low_below_band_any.mean()) if len(subset) else None,
        'note': 'Plausibility windows are exploratory, not validated component classifications.',
    }
    stars.reset_index().to_csv(OUTPUT_DIR / 'per_star_stability.csv', index=False)
    summary = {'overall': overall, 'component_stability': stats,
               'paired_component_comparisons': paired, 'holdout_by_filter': hold,
               'plausibility_windows': {'mid_nu_over_numax': MID_NU_MAX_RANGE,
                 'high_nu_over_numax': HIGH_NU_MAX_RANGE,
                 'mid_over_high_frequency': MID_HIGH_RATIO_RANGE}}
    write_json(OUTPUT_DIR / 'summary.json', summary)

    # Summary table, readable without opening JSON.
    rows = []
    for key, d in stats.items():
        rows.append({'component_metric': key, **d})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'component_stability_summary.csv', index=False)

    # Figure 1: spread distributions, independent stars.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, kind, title in zip(axes, ('freq', 'rms'), ('Characteristic frequency', 'Integrated RMS amplitude')):
        for model, comp, label in (('two_fixed', 'long', '2-component long'),
                                    ('two_fixed', 'short', '2-component short'),
                                    ('three_fixed', 'mid', '3-component mid'),
                                    ('three_fixed', 'high', '3-component high')):
            x = stars[f'{model}_{comp}_{kind}_range_fraction'].dropna()
            x = x[np.isfinite(x) & (x >= 0)]
            if len(x):
                ax.plot(np.linspace(0, 1.0, 200),
                        [(x <= t).mean() for t in np.linspace(0, 1.0, 200)], label=f'{label} (n={len(x)})')
        ax.axvline(.10, color='grey', linestyle='--', linewidth=1)
        ax.set(xlabel='Max/min across filters − 1', ylabel='Fraction of stars', title=title,
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'stability_cdfs.png', dpi=180)
    plt.close(fig)

    # Figure 2: physical ratios by filter and a simple stability/plausibility view.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, comp, xlim in zip(axes[:2], ('mid', 'high'), ((0, 1.2), (0, 2.2))):
        data = [stars[f'three_fixed_{comp}_nu_over_numax_{filt}'].replace([np.inf,-np.inf], np.nan).dropna()
                for filt in FILTERS]
        ax.boxplot(data, tick_labels=FILTERS, showfliers=False)
        ax.set(title=rf'Three-fixed {comp} / $\nu_{{\max}}$', ylabel='Frequency ratio')
        ax.grid(alpha=.2)
    axes[2].scatter(stars.three_fixed_mid_freq_range_fraction,
                    stars.three_fixed_mid_rms_range_fraction,
                    c=stars.plausible_all_three.astype(int), cmap='coolwarm', alpha=.65, s=17)
    axes[2].axvline(.1, linestyle='--', color='grey')
    axes[2].axhline(.1, linestyle='--', color='grey')
    axes[2].set(xlabel='Mid frequency max/min − 1', ylabel='Mid RMS max/min − 1',
                title='Mid stability (red: plausible all filters)', xlim=(0, 1), ylim=(0, 1))
    axes[2].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'physical_ratios_and_stability.png', dpi=180)
    plt.close(fig)

    # Figure 3: 20d versus 80d frequency and RMS (log axes).
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, kind in zip(axes, ('freq', 'rms')):
        x = stars[f'three_fixed_mid_{kind}_20d']
        y = stars[f'three_fixed_mid_{kind}_80d']
        mask = x.gt(0) & y.gt(0) & x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], alpha=.4, s=16)
        if mask.any():
            lo = min(x[mask].min(), y[mask].min())
            hi = max(x[mask].max(), y[mask].max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set(xscale='log', yscale='log', xlabel=f'20d mid {kind}',
               ylabel=f'80d mid {kind}', title=f'Mid component: {kind}')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'mid_20d_vs_80d.png', dpi=180)
    plt.close(fig)

    print('\n=== V4 CROSS-FILTER STABILITY ===')
    print(json.dumps(overall, indent=2))
    print('\nComponent stability (median max/min - 1):')
    for k, v in stats.items():
        print(f'  {k:28s} n={v["n_complete_stars"]:3d}  median={v["median_max_over_min_minus_1"]}')
    print('\nPaired old/new comparisons:')
    print(json.dumps(paired, indent=2))
    print('\nHoldout by filter:')
    print(json.dumps(hold, indent=2))
    print('\nSaved to:', OUTPUT_DIR)

if __name__ == '__main__':
    main()
