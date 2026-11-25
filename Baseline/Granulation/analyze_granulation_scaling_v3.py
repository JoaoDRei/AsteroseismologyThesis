#!/usr/bin/env python3
"""V3 granulation scaling and matched-model sensitivity analysis. Edit CONFIG only."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr

# ===================== CONFIGURATION =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if script is elsewhere
MANIFEST = PROJECT_ROOT / 'Baseline' / 'manifest.csv'
ANALYSIS = 'full'  # 'full' or 'pilot'
FIT_FILES = {
    'full': {
        'two_fixed': PROJECT_ROOT / 'data/fitted_parameters_v3_two_fixed_all/fit_results.csv',
        'two_free': PROJECT_ROOT / 'data/fitted_parameters_v3_two_free_all/fit_results.csv',
    },
    'pilot': {
        'two_fixed': PROJECT_ROOT / 'data/fitted_parameters_v3_pilot_300/fit_results.csv',
        'two_free': PROJECT_ROOT / 'data/fitted_parameters_v3_pilot_300/fit_results.csv',
    },
}
OUTPUT_DIR = PROJECT_ROOT / 'results' / f'granulation_scaling_v3_{ANALYSIS}'
DURATIONS = ('20d', '55d', '80d')
PRIMARY_DURATION = '55d'
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42
MIN_REGRESSION_N = 30
MAKE_PLOTS = True
# "all_valid" retains boundary fits; "no_bounds" removes alpha/width-bound fits.
SAMPLES = ('all_valid', 'no_bounds')
TARGETS = {
    'short_component_nu_char_uHz': 'Short characteristic frequency (uHz)',
    'short_integrated_rms_native_sqrt_uHz': 'Short integrated amplitude (native units)',
    'long_component_nu_char_uHz': 'Long characteristic frequency (uHz)',
    'long_integrated_rms_native_sqrt_uHz': 'Long integrated amplitude (native units)',
}
FREE_ONLY_TARGETS = {'short_component_alpha': 'Short exponent', 'long_component_alpha': 'Long exponent'}
# =========================================================


def clean_numeric(series):
    s = pd.to_numeric(series, errors='coerce')
    return s.mask(~np.isfinite(s) | (s <= -999))


def metadata_table(path):
    df = pd.read_csv(path, low_memory=False)
    needed = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate']
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f'Manifest missing: {sorted(missing)}')
    df = df[needed].copy()
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic'])
    df['kic'] = df.kic.astype('int64')
    for c in ('nu_max', 'teff', 'metallicity'):
        df[c] = clean_numeric(df[c])
    df.loc[df.nu_max <= 0, 'nu_max'] = np.nan
    df.loc[df.teff <= 0, 'teff'] = np.nan
    df['evolstate'] = df.evolstate.astype('string').str.strip().str.upper()
    for c in ('nu_max', 'teff', 'metallicity', 'evolstate'):
        conflicts = (df.groupby('kic')[c].nunique(dropna=True) > 1).sum()
        if conflicts:
            warnings.warn(f'{conflicts} KICs have conflicting {c} values across manifest rows')
    numeric = df.groupby('kic', as_index=False)[['nu_max', 'teff', 'metallicity']].median()
    states = df.groupby('kic')['evolstate'].agg(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else pd.NA
    ).reset_index()
    return numeric.merge(states, on='kic', validate='one_to_one')


def read_model(path, model):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {model} results: {path}')
    df = pd.read_csv(path, low_memory=False)
    required = {'kic', 'duration', *TARGETS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    if 'model' in df:
        df = df[df.model.astype(str).str.strip() == model].copy()
    else:
        raise ValueError(f'{path} has no model column: expected V3')
    if df.empty:
        raise ValueError(f'No {model} rows in {path}')
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic'])
    df['kic'] = df.kic.astype('int64')
    df['duration'] = df.duration.astype(str).str.strip().str.lower().str.replace('days', 'd', regex=False)
    df['duration'] = df.duration.str.replace(r'^(20|55|80)$', r'\1d', regex=True)
    df = df[df.duration.isin(DURATIONS)].copy()
    if 'success' in df:
        df = df[df.success.astype(str).str.lower().isin(('true', '1'))].copy()
    if df.duplicated(['kic', 'duration']).any():
        raise ValueError(f'{model}: duplicate KIC/duration after model selection')
    for c in list(TARGETS) + list(FREE_ONLY_TARGETS) + ['oscillation_envelope_log_width', 'rms_log10_residual']:
        if c in df:
            df[c] = clean_numeric(df[c])
    df['alpha_bound'] = False
    if model == 'two_free':
        for c in FREE_ONLY_TARGETS:
            if c in df:
                df['alpha_bound'] |= np.isclose(df[c], 2, atol=.01, rtol=0) | np.isclose(df[c], 6, atol=.01, rtol=0)
    df['width_bound'] = False
    if 'oscillation_envelope_log_width' in df:
        w = df.oscillation_envelope_log_width
        df['width_bound'] = np.isclose(w, .08, atol=.01, rtol=0) | np.isclose(w, .8, atol=.01, rtol=0)
    df['any_bound'] = df.alpha_bound | df.width_bound
    df['model'] = model
    return df


def design(df, target):
    cols = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate', target]
    d = df[cols].dropna().copy()
    d = d[(d.nu_max > 0) & (d.teff > 0) & (d[target] > 0) & d.evolstate.isin(['RGB', 'RC'])].copy()
    d['log_nu'] = np.log10(d.nu_max / 40.)
    d['log_teff'] = np.log10(d.teff / 4800.)
    d['is_rc'] = (d.evolstate == 'RC').astype(float)
    d['log_y'] = np.log10(d[target])
    return d


def fit_ols(d, metallicity=True):
    names = ['intercept', 'log10(nu_max/40uHz)', 'log10(teff/4800K)', 'RC_indicator']
    columns = [np.ones(len(d)), d.log_nu.to_numpy(), d.log_teff.to_numpy(), d.is_rc.to_numpy()]
    if metallicity:
        columns.append(d.metallicity.to_numpy())
        names.append('[Fe/H]')
    X = np.column_stack(columns).astype(float)
    y = d.log_y.to_numpy(dtype=float)
    n, p = X.shape
    if n < MIN_REGRESSION_N or n <= p + 2 or np.linalg.matrix_rank(X) != p:
        raise ValueError('Insufficient observations or rank-deficient design')
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    inv = np.linalg.inv(X.T @ X)
    leverage = np.einsum('ij,jk,ik->i', X, inv, X)
    hc3 = (resid / np.maximum(1 - leverage, 1e-8)) ** 2
    cov = inv @ (X.T @ (X * hc3[:, None])) @ inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * norm.sf(np.abs(z))
    ss = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan
    coefficients = {name: {'estimate': float(b), 'hc3_se': float(s),
                           'ci95_lower': float(b - 1.96 * s), 'ci95_upper': float(b + 1.96 * s),
                           'p_value_normal_approx': float(pv)}
                    for name, b, s, pv in zip(names, beta, se, pval)}
    return {'n': n, 'r_squared': r2, 'coefficients': coefficients}, resid


def regress(d, target):
    t = design(d, target)
    baseline, resid = fit_ols(t, metallicity=False)
    extended, _ = fit_ols(t, metallicity=True)
    rho = spearmanr(t.metallicity.to_numpy(), resid) if len(t) >= 4 else None
    return t, {'baseline': baseline, 'with_metallicity': extended,
               'delta_r_squared': extended['r_squared'] - baseline['r_squared'],
               'baseline_residual_spearman': {'rho': float(rho.statistic), 'p': float(rho.pvalue)} if rho else None}


def plot_target(d, target, model, duration, out):
    t = design(d, target)
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for state, color in [('RGB', 'tab:blue'), ('RC', 'tab:orange')]:
        s = t[t.evolstate == state]
        axes[0].scatter(s.nu_max, s[target], s=8, alpha=.35, color=color, label=f'{state} ({len(s)})', rasterized=True)
        axes[1].scatter(s.metallicity, s.log_y, s=8, alpha=.35, color=color, label=state, rasterized=True)
    axes[0].set(xscale='log', yscale='log', xlabel=r'Catalogue $\nu_{\max}$ ($\mu$Hz)', ylabel=TARGETS.get(target, target))
    axes[1].set(xlabel='[Fe/H] (dex)', ylabel=f'log10({target})')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(f'{model} | {duration}')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def paired_difference(a, b, target, rng):
    """Paired stellar bootstrap for coefficient (two_free minus two_fixed)."""
    da = design(a, target)
    db = design(b, target)
    common = da.merge(db, on='kic', suffixes=('_fixed', '_free'), validate='one_to_one')
    # Reuse exactly the same covariates in both regressions.
    if common.empty:
        raise ValueError('No matched stars')
    for col in ['log_nu', 'log_teff', 'is_rc', 'metallicity']:
        if not np.allclose(common[f'{col}_fixed'], common[f'{col}_free'], rtol=0, atol=1e-10):
            raise ValueError(f'Conflicting paired covariates: {col}')
    X = np.column_stack([np.ones(len(common)), common.log_nu_fixed, common.log_teff_fixed,
                         common.is_rc_fixed, common.metallicity_fixed])
    y0 = common.log_y_fixed.to_numpy()
    y1 = common.log_y_free.to_numpy()
    if len(common) < MIN_REGRESSION_N or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError('Too few matched stars or rank-deficient design')
    b0 = np.linalg.lstsq(X, y0, rcond=None)[0][-1]
    b1 = np.linalg.lstsq(X, y1, rcond=None)[0][-1]
    diffs = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(common), len(common))
        xx = X[idx]
        if np.linalg.matrix_rank(xx) < xx.shape[1]:
            continue
        diffs.append(float(np.linalg.lstsq(xx, y1[idx] - y0[idx], rcond=None)[0][-1]))
    if not diffs:
        raise ValueError('No valid bootstrap resamples')
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'n_matched': len(common), 'feh_beta_two_fixed': float(b0),
            'feh_beta_two_free': float(b1), 'difference_free_minus_fixed': float(b1-b0),
            'bootstrap_ci95_lower': float(lo), 'bootstrap_ci95_upper': float(hi),
            'n_bootstrap_valid': len(diffs)}


def main():
    if ANALYSIS not in FIT_FILES:
        raise ValueError(f'Unknown ANALYSIS={ANALYSIS}')
    if PRIMARY_DURATION not in DURATIONS:
        raise ValueError('Invalid PRIMARY_DURATION')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = metadata_table(MANIFEST)
    models = {}
    for model, path in FIT_FILES[ANALYSIS].items():
        fits = read_model(path, model)
        merged = fits.merge(meta, on='kic', how='left', validate='many_to_one', indicator=True)
        merged['valid_stellar_metadata'] = (merged.nu_max.gt(0) & merged.teff.gt(0) &
                                            merged.metallicity.notna() & merged.evolstate.isin(['RGB', 'RC']))
        merged.to_csv(OUTPUT_DIR / f'{model}_all_durations_merged.csv', index=False)
        models[model] = merged
        print(f'{model}: {len(fits)} fits, {fits.kic.nunique()} unique KICs; '
              f'{(merged._merge == "left_only").sum()} unmatched metadata rows')
    summary = []
    reports = {}
    for model, merged in models.items():
        reports[model] = {}
        for duration in DURATIONS:
            d = merged[(merged.duration == duration) & merged.valid_stellar_metadata].copy()
            reports[model][duration] = {'n_valid': len(d), 'n_bound': int(d.any_bound.sum())}
            for sample in SAMPLES:
                sub = d if sample == 'all_valid' else d[~d.any_bound].copy()
                sub.to_csv(OUTPUT_DIR / f'{model}_{duration}_{sample}_stars.csv', index=False)
                targets = dict(TARGETS)
                if model == 'two_free':
                    targets.update(FREE_ONLY_TARGETS)
                for target in targets:
                    try:
                        t, result = regress(sub, target)
                        c = result['with_metallicity']['coefficients']['[Fe/H]']
                        summary.append({'model': model, 'duration': duration, 'sample': sample, 'target': target,
                                        'n': len(t), 'feh_coefficient': c['estimate'], 'feh_hc3_se': c['hc3_se'],
                                        'feh_ci95_lower': c['ci95_lower'], 'feh_ci95_upper': c['ci95_upper'],
                                        'feh_p_value': c['p_value_normal_approx'],
                                        'baseline_r2': result['baseline']['r_squared'],
                                        'extended_r2': result['with_metallicity']['r_squared'],
                                        'delta_r2': result['delta_r_squared'], 'error': ''})
                        if MAKE_PLOTS and sample == 'all_valid':
                            plot_target(sub, target, model, duration, OUTPUT_DIR / f'{model}_{duration}_{target}.png')
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        summary.append({'model': model, 'duration': duration, 'sample': sample, 'target': target,
                                        'n': len(design(sub, target)), 'error': str(exc)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'regression_summary.csv', index=False)
    paired_rows = []
    if 'two_fixed' in models and 'two_free' in models:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for duration in DURATIONS:
            for sample in SAMPLES:
                a = models['two_fixed']
                b = models['two_free']
                a = a[(a.duration == duration) & a.valid_stellar_metadata].copy()
                b = b[(b.duration == duration) & b.valid_stellar_metadata].copy()
                # Common quality selection: for no_bounds, retain only stars passing BOTH models.
                if sample == 'no_bounds':
                    a = a[~a.any_bound]
                    b = b[~b.any_bound]
                for target in TARGETS:
                    try:
                        res = paired_difference(a, b, target, rng)
                        paired_rows.append({'duration': duration, 'sample': sample, 'target': target,
                                            **res, 'error': ''})
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        paired_rows.append({'duration': duration, 'sample': sample, 'target': target,
                                            'error': str(exc)})
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUTPUT_DIR / 'paired_model_comparison.csv', index=False)
    config = {'analysis': ANALYSIS, 'manifest': str(MANIFEST),
              'fit_files': {k: str(v) for k, v in FIT_FILES[ANALYSIS].items()},
              'primary_duration': PRIMARY_DURATION, 'bootstrap_reps': BOOTSTRAP_REPS,
              'bootstrap_seed': BOOTSTRAP_SEED, 'sample_counts': reports,
              'caveats': ['Durations are not independent observations.',
                          'HC3 is heteroskedasticity-robust but does not model errors in covariates.',
                          'The paired bootstrap resamples unique KICs within each duration.',
                          'Native integrated amplitudes require PSD normalization verification.',
                          'No-bound selection may induce bias.',
                          'Exploratory associations, not causal estimates.']}
    (OUTPUT_DIR / 'analysis_summary.json').write_text(json.dumps(config, indent=2, default=str))
    lines = [f'V3 GRANULATION SCALING: {ANALYSIS}', f'Primary duration: {PRIMARY_DURATION}',
             'OLS HC3 SE; paired star-level bootstrap for free-minus-fixed metallicity coefficient.',
             'All three filter durations share the same underlying observations.', '']
    show = summary_df[(summary_df.duration == PRIMARY_DURATION) &
                      (summary_df.target.isin(TARGETS))]
    cols = ['model', 'sample', 'target', 'n', 'feh_coefficient', 'feh_hc3_se', 'feh_ci95_lower', 'feh_ci95_upper', 'error']
    lines.append('INDIVIDUAL REGRESSIONS (primary duration):')
    lines.append(show.reindex(columns=cols).to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    if not paired_df.empty:
        showp = paired_df[paired_df.duration == PRIMARY_DURATION]
        lines += ['', 'PAIRED MODEL DIFFERENCES (two_free minus two_fixed):',
                  showp.to_string(index=False, float_format=lambda x: f'{x:.4g}')]
    (OUTPUT_DIR / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved to: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
