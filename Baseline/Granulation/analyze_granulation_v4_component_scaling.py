#!/usr/bin/env python3
"""V4 three-Harvey-component physical-ratio diagnostics (NO PSD refitting).

Run from project root:
    python -m Baseline.analysis.analyze_granulation_v4_component_scaling

Place in Baseline/analysis/ or adjust PROJECT_ROOT. Reads the V4 fit_results.csv.
The 20d/55d/80d labels are KADACS high-pass FILTER timescales applied
 to largely the same four-year Kepler light curves, NOT independent durations.

Examines mid/nu_max, high/nu_max, mid/high frequency and integrated RMS
ratios, separately per filter and with one summary per KIC. All ratios are
computed within the same fitted spectrum. Native RMS units are not assumed
 to be ppm. This is descriptive analysis, not a physical component classifier.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================== CONFIG ==================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIT_CSV = PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_all/fit_results.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/granulation_v4_component_scaling_full'
FILTERS = ('20d', '55d', '80d')
MODEL = 'three_fixed'
PRIMARY_FILTER = '55d'
N_NU_MAX_BINS = 6
MIN_BIN_STARS = 8
BOOTSTRAP_REPS = 1000  # Resample stars, not correlated filter spectra
BOOTSTRAP_SEED = 42
MAKE_PLOTS = True
PLOT_DPI = 170
# Exploratory windows: descriptive only; NOT validated identification criteria.
WINDOWS = {
    'mid_over_numax': (0.15, 0.60),
    'high_over_numax': (0.50, 1.80),
    'mid_over_high_frequency': (0.20, 0.50),
}
# ========================================================================

RATIO_COLUMNS = (
    'mid_over_numax', 'high_over_numax', 'mid_over_high_frequency',
    'mid_over_high_rms', 'low_over_numax', 'low_over_mid_frequency',
)
LABELS = {
    'mid_over_numax': r'$b_{\rm mid}/\nu_{\max}$',
    'high_over_numax': r'$b_{\rm high}/\nu_{\max}$',
    'mid_over_high_frequency': r'$b_{\rm mid}/b_{\rm high}$',
    'mid_over_high_rms': r'$a_{\rm mid}/a_{\rm high}$',
    'low_over_numax': r'$b_{\rm low}/\nu_{\max}$',
    'low_over_mid_frequency': r'$b_{\rm low}/b_{\rm mid}$',
}


def positive(s):
    x = pd.to_numeric(s, errors='coerce')
    return x.where(np.isfinite(x) & (x > 0))


def true_flags(s):
    return s.astype('string').str.strip().str.lower().isin(('true', '1', 'yes'))


def as_float(x):
    return float(x) if np.isfinite(x) else None


def clean_ratio(a, b):
    z = a / b
    return z.where(np.isfinite(z) & (z > 0))


def read_fits():
    if not FIT_CSV.is_file():
        raise FileNotFoundError(f'Cannot find {FIT_CSV}. Change FIT_CSV in CONFIG.')
    df = pd.read_csv(FIT_CSV, low_memory=False)
    cols = ['kic', 'duration', 'model', 'success', 'nu_max_catalogue']
    for comp in ('low', 'mid', 'high'):
        cols += [f'{comp}_component_nu_char_uHz',
                 f'{comp}_integrated_rms_native_sqrt_uHz']
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required V4 columns: {missing}')
    df = df[true_flags(df['success']) & (df['model'].astype(str).str.strip() == MODEL)].copy()
    df['duration'] = df['duration'].astype(str).str.lower().str.strip()
    df = df[df.duration.isin(FILTERS)].copy()
    df['kic'] = pd.to_numeric(df['kic'], errors='coerce')
    df = df.dropna(subset=['kic'])
    if df.empty:
        raise ValueError('No successful three_fixed fits for the requested filters.')
    if not np.all(df.kic == np.floor(df.kic)):
        raise ValueError('Non-integer KIC values encountered.')
    df['kic'] = df.kic.astype('int64')
    if df.duplicated(['kic', 'duration']).any():
        example = df.loc[df.duplicated(['kic', 'duration'], keep=False),
                         ['kic', 'duration']].head(10).to_dict('records')
        raise ValueError(f'Duplicate KIC/filter/model rows: {example}')
    for col in cols:
        if col in ('kic', 'duration', 'model', 'success'):
            continue
        df[col] = positive(df[col])
    for comp in ('low', 'mid', 'high'):
        for flag in ('char_below_fit_band', 'char_below_filter_timescale',
                     'char_near_parameter_bound'):
            col = f'{comp}_{flag}'
            df[col] = true_flags(df[col]) if col in df else False

    nu = df['nu_max_catalogue']
    lo = df['low_component_nu_char_uHz']
    mi = df['mid_component_nu_char_uHz']
    hi = df['high_component_nu_char_uHz']
    ar = df['mid_integrated_rms_native_sqrt_uHz']
    br = df['high_integrated_rms_native_sqrt_uHz']
    df['mid_over_numax'] = clean_ratio(mi, nu)
    df['high_over_numax'] = clean_ratio(hi, nu)
    df['mid_over_high_frequency'] = clean_ratio(mi, hi)
    df['mid_over_high_rms'] = clean_ratio(ar, br)
    df['low_over_numax'] = clean_ratio(lo, nu)
    df['low_over_mid_frequency'] = clean_ratio(lo, mi)
    df['mid_high_separation_factor'] = clean_ratio(hi, mi)
    df['low_mid_separation_factor'] = clean_ratio(mi, lo)
    df['low_below_band'] = df['low_char_below_fit_band']
    df['mid_below_band'] = df['mid_char_below_fit_band']
    df['high_below_band'] = df['high_char_below_fit_band']
    df['all_windows'] = True
    for col, (lower, upper) in WINDOWS.items():
        df[f'{col}_in_window'] = df[col].between(lower, upper, inclusive='both')
        df['all_windows'] &= df[f'{col}_in_window']
    return df.sort_values(['kic', 'duration'])


def quantiles(x):
    a = pd.to_numeric(pd.Series(x), errors='coerce').to_numpy(float)
    a = a[np.isfinite(a) & (a > 0)]
    if not len(a):
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, p10=np.nan,
                    p16=np.nan, p84=np.nan, p90=np.nan, iqr=np.nan,
                    log10_scatter_mad=np.nan)
    med_log = np.median(np.log10(a))
    return dict(n=len(a), median=float(np.median(a)), mean=float(np.mean(a)),
                std=float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                p10=float(np.percentile(a, 10)), p16=float(np.percentile(a, 16)),
                p84=float(np.percentile(a, 84)), p90=float(np.percentile(a, 90)),
                iqr=float(np.percentile(a, 75)-np.percentile(a, 25)),
                log10_scatter_mad=float(1.4826*np.median(np.abs(np.log10(a)-med_log))))


def bootstrap_median_ci(x, rng):
    a = pd.to_numeric(pd.Series(x), errors='coerce').to_numpy(float)
    a = a[np.isfinite(a) & (a > 0)]
    if not len(a) or BOOTSTRAP_REPS <= 0:
        return np.nan, np.nan
    meds = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        meds[i] = np.median(a[rng.integers(0, len(a), len(a))])
    return tuple(float(v) for v in np.percentile(meds, [2.5, 97.5]))


def per_star(df):
    # One row per KIC; medians across filters describe each star once.
    # These filters are correlated, so do not treat them as three replicates.
    rows = []
    for kic, g in df.groupby('kic', sort=True):
        row = {'kic': int(kic), 'n_filters': len(g),
               'filters': ','.join(x for x in FILTERS if x in set(g.duration)),
               'nu_max_catalogue': float(g.nu_max_catalogue.median()),
               'low_below_band_any': bool(g.low_below_band.any()),
               'mid_below_band_any': bool(g.mid_below_band.any()),
               'high_below_band_any': bool(g.high_below_band.any()),
               'all_windows_all_filters': bool(g.all_windows.all()),
               'all_windows_any_filter': bool(g.all_windows.any())}
        for col in RATIO_COLUMNS:
            vals = g[col].dropna().to_numpy(float)
            row[col] = float(np.median(vals)) if len(vals) else np.nan
            row[col+'_max_over_min'] = float(vals.max()/vals.min()) if len(vals) >= 2 else np.nan
        for comp in ('mid', 'high'):
            for suffix in ('component_nu_char_uHz', 'integrated_rms_native_sqrt_uHz'):
                col = f'{comp}_{suffix}'
                vals = g[col].dropna().to_numpy(float)
                row[col+'_max_over_min'] = float(vals.max()/vals.min()) if len(vals) >= 2 else np.nan
        for col, (lower, upper) in WINDOWS.items():
            row[col+'_in_window_median'] = bool(lower <= row[col] <= upper) if np.isfinite(row[col]) else False
        row['all_windows_median'] = all(row[col+'_in_window_median'] for col in WINDOWS)
        rows.append(row)
    return pd.DataFrame(rows)


def summaries(df, stars):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name, group in [(f, df[df.duration == f]) for f in FILTERS] + [('per_star_median', stars)]:
        for col in RATIO_COLUMNS:
            stats = quantiles(group[col])
            lo, hi = bootstrap_median_ci(group[col], rng)
            row = {'group': name, 'ratio': col, **stats,
                   'median_ci95_low_star_bootstrap': lo,
                   'median_ci95_high_star_bootstrap': hi}
            if col in WINDOWS:
                lower, upper = WINDOWS[col]
                valid = group[col].notna()
                row['fraction_in_exploratory_window'] = (float(group.loc[valid, col].between(lower, upper).mean())
                                                         if valid.any() else np.nan)
            else:
                row['fraction_in_exploratory_window'] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def binned_summary(df, stars):
    rows = []
    # Identical bin edges for all filters, defined on unique stars.
    valid = stars[stars.nu_max_catalogue.gt(0)].copy()
    if len(valid) < 2:
        return pd.DataFrame()
    edges = np.unique(np.quantile(np.log10(valid.nu_max_catalogue),
                                  np.linspace(0, 1, N_NU_MAX_BINS + 1)))
    if len(edges) < 2:
        return pd.DataFrame()
    for label, group in [(f, df[df.duration == f]) for f in FILTERS] + [('per_star_median', stars)]:
        g = group[group.nu_max_catalogue.gt(0)].copy()
        g['nu_bin'] = pd.cut(np.log10(g.nu_max_catalogue), edges,
                             include_lowest=True, duplicates='drop')
        for bin_label, part in g.groupby('nu_bin', observed=True):
            if len(part) < MIN_BIN_STARS:
                continue
            for col in RATIO_COLUMNS[:4]:
                row = {'group': label, 'ratio': col, 'n_stars': len(part),
                       'nu_max_bin_low': float(10**bin_label.left),
                       'nu_max_bin_high': float(10**bin_label.right),
                       'median_nu_max': float(part.nu_max_catalogue.median()),
                       **quantiles(part[col])}
                rows.append(row)
    return pd.DataFrame(rows)


def plot_ratio_panels(df, stars, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {'20d': 'tab:blue', '55d': 'tab:orange', '80d': 'tab:green'}
    for ax, col in zip(axes.flat, RATIO_COLUMNS[:4]):
        allvals = df[col].dropna().to_numpy(float)
        allvals = allvals[np.isfinite(allvals) & (allvals > 0)]
        if len(allvals) < 2:
            continue
        low, high = np.percentile(np.log10(allvals), [1, 99])
        if high <= low:
            high = low + 0.1
        edges = np.logspace(low, high, 36)
        for filt in FILTERS:
            vals = df.loc[df.duration == filt, col].dropna().to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                ax.hist(vals, bins=edges, density=False, histtype='step',
                        lw=1.8, label=filt, color=colors[filt])
        if col in WINDOWS:
            lower, upper = WINDOWS[col]
            ax.axvspan(lower, upper, alpha=.08, color='grey', label='Exploratory window')
        med = stars[col].median()
        if np.isfinite(med):
            ax.axvline(med, color='black', linestyle='--', lw=1, label='Star median')
        ax.set_xscale('log')
        ax.set(xlabel=LABELS[col], ylabel='Number of spectra', title=col.replace('_', ' '))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Three-fixed: dimensionless component ratios by KADACS filter')
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_vs_numax(df, stars, binned, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {'20d': 'tab:blue', '55d': 'tab:orange', '80d': 'tab:green'}
    for ax, col in zip(axes.flat, RATIO_COLUMNS[:4]):
        for filt in FILTERS:
            g = df[df.duration == filt]
            valid = g.nu_max_catalogue.gt(0) & g[col].gt(0)
            ax.scatter(g.loc[valid, 'nu_max_catalogue'], g.loc[valid, col],
                       s=8, alpha=.16, color=colors[filt], rasterized=True, label=filt)
        q = binned[(binned.group == 'per_star_median') & (binned.ratio == col)] if not binned.empty else pd.DataFrame()
        if not q.empty:
            ax.plot(q.median_nu_max, q['median'], color='black', marker='o',
                    markersize=4, lw=2, label='Per-star binned median')
            ax.fill_between(q.median_nu_max.to_numpy(float), q.p16.to_numpy(float),
                            q.p84.to_numpy(float), color='grey', alpha=.15)
        if col in WINDOWS:
            ax.axhspan(*WINDOWS[col], color='grey', alpha=.08)
        ax.set(xscale='log', yscale='log', xlabel=r'Catalogue $\nu_{\max}$ ($\mu$Hz)',
               ylabel=LABELS[col])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Component ratios versus catalogue frequency (correlated filter estimates)')
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_per_star_filter_spread(stars, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, col in zip(axes, ('mid_over_high_frequency', 'mid_over_high_rms')):
        x = stars.nu_max_catalogue.to_numpy(float)
        y = stars[col].to_numpy(float)
        spread = stars[col+'_max_over_min'].to_numpy(float)
        valid = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0) & np.isfinite(spread)
        sc = ax.scatter(x[valid], y[valid], c=np.clip(spread[valid]-1, 0, 1),
                        cmap='viridis', vmin=0, vmax=1, s=22, alpha=.8)
        fig.colorbar(sc, ax=ax, label='Filter max/min − 1 (clipped at 1)')
        ax.set(xscale='log', yscale='log', xlabel=r'Catalogue $\nu_{\max}$ ($\mu$Hz)',
               ylabel=LABELS[col])
        ax.grid(alpha=.2)
    fig.suptitle('Per-star median ratios and cross-filter variation')
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def main():
    if PRIMARY_FILTER not in FILTERS:
        raise ValueError('PRIMARY_FILTER must be one of FILTERS')
    if N_NU_MAX_BINS < 2:
        raise ValueError('N_NU_MAX_BINS must be >=2')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_fits()
    stars = per_star(df)
    summary = summaries(df, stars)
    bins = binned_summary(df, stars)
    df.to_csv(OUTPUT_DIR / 'per_spectrum_ratios.csv', index=False)
    stars.to_csv(OUTPUT_DIR / 'per_star_ratios.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'ratio_summary.csv', index=False)
    bins.to_csv(OUTPUT_DIR / 'numax_binned_ratios.csv', index=False)

    window_rows = []
    for label, g in [(f, df[df.duration == f]) for f in FILTERS] + [('per_star_median', stars)]:
        row = {'group': label, 'n_stars': len(g)}
        for col in WINDOWS:
            flag = col + ('_in_window_median' if label == 'per_star_median' else '_in_window')
            row[col+'_in_window_fraction'] = float(g[flag].mean()) if len(g) else np.nan
        row['all_windows_fraction'] = float(g['all_windows_median' if label == 'per_star_median' else 'all_windows'].mean()) if len(g) else np.nan
        row['low_below_band_fraction'] = float(g['low_below_band_any' if label == 'per_star_median' else 'low_below_band'].mean()) if len(g) else np.nan
        window_rows.append(row)
    pd.DataFrame(window_rows).to_csv(OUTPUT_DIR / 'exploratory_window_counts.csv', index=False)

    if MAKE_PLOTS:
        plot_ratio_panels(df, stars, OUTPUT_DIR / 'ratio_distributions.png')
        plot_vs_numax(df, stars, bins, OUTPUT_DIR / 'ratios_vs_numax.png')
        plot_per_star_filter_spread(stars, OUTPUT_DIR / 'per_star_ratio_filter_spread.png')

    primary = summary[(summary.group == PRIMARY_FILTER) & summary.ratio.isin(RATIO_COLUMNS[:4])]
    star_med = summary[(summary.group == 'per_star_median') & summary.ratio.isin(RATIO_COLUMNS[:4])]
    lines = [f'V4 THREE-FIXED COMPONENT RATIO DIAGNOSTICS',
             f'Input: {FIT_CSV.resolve()}',
             f'Unique KICs: {len(stars)}; successful spectra: {len(df)}',
             f'All three filters: {(stars.n_filters == len(FILTERS)).sum()} stars',
             'IMPORTANT: KADACS 20d/55d/80d are high-pass filter timescales, not independent datasets.',
             'RMS amplitude ratios are dimensionless, but native absolute RMS units are NOT assumed to be ppm.',
             'Ratios use components sorted by fitted frequency; physical identity is not established.',
             'Window fractions are exploratory, not validated mesogranulation detection rates.', '',
             f'PRIMARY FILTER: {PRIMARY_FILTER}',
             primary[['ratio','n','median','p16','p84','log10_scatter_mad',
                      'fraction_in_exploratory_window']].to_string(index=False, float_format=lambda x: f'{x:.4g}'),
             '', 'ONE OBSERVATION PER STAR (median across available filters)',
             star_med[['ratio','n','median','p16','p84','log10_scatter_mad',
                       'fraction_in_exploratory_window']].to_string(index=False, float_format=lambda x: f'{x:.4g}'),
             '', 'WINDOW / LOW-FREQUENCY FLAGS',
             pd.DataFrame(window_rows).to_string(index=False, float_format=lambda x: f'{x:.3f}')]
    (OUTPUT_DIR / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    config = {'input': str(FIT_CSV.resolve()), 'model': MODEL,
              'n_unique_kics': len(stars), 'n_successful_spectra': len(df),
              'filters': list(FILTERS), 'primary_filter': PRIMARY_FILTER,
              'windows': WINDOWS, 'bootstrap_reps': BOOTSTRAP_REPS,
              'nu_max_bins': N_NU_MAX_BINS,
              'caveats': ['Filters share underlying observations; bootstrap only resamples stars.',
                          'Frequency ordering is not physical identification.',
                          'Integrated RMS can be extrapolation-dominated, especially below fit band.',
                          'Amplitude ratio is native-unit invariant only if both components use same PSD normalization.',
                          'Windows are exploratory; this is not a Corsaro model replication.']}
    (OUTPUT_DIR / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nSaved to {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
