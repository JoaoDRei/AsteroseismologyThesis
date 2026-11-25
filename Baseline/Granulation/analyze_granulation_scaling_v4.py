#!/usr/bin/env python3
"""V3/V4 Harvey scaling, quality sensitivity and matched-star comparisons.

Edit CONFIG only. Run from project root:
    python -m Baseline.analysis.analyze_granulation_scaling_v4

KADACS 20d/55d/80d designate high-pass filters of the SAME ~4-year Kepler
observations, not independent observing durations. Fits are never repeated.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr

# =========================== CONFIG ===========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / 'Baseline' / 'manifest.csv'
ANALYSIS = 'full'  # 'pilot' or 'full'; full requires a V4 full fit CSV
FIT_FILES = {
    'pilot': {
        'two_fixed': PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_pilot/fit_results.csv',
        'three_fixed': PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_pilot/fit_results.csv',
        # To compare against the ORIGINAL V3 pilot instead, change two_fixed path above:
        # PROJECT_ROOT / 'data/fitted_parameters_v3_pilot_300/fit_results.csv'
    },
    'full': {
        'two_fixed': PROJECT_ROOT / 'data/fitted_parameters_v3_two_fixed_all/fit_results.csv',
        'three_fixed': PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_all/fit_results.csv',
    },
}
OUTPUT_DIR = PROJECT_ROOT / 'results' / f'granulation_scaling_v4_{ANALYSIS}'
FILTERS = ('20d', '55d', '80d')
PRIMARY_FILTER = '55d'
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42
MIN_REGRESSION_N = 30
MAKE_PLOTS = True
# Selection criteria are diagnostic, NOT confirmed component classifications.
SAMPLES = ('all_valid', 'no_bounds', 'stable_mid_10pct', 'stable_mid_plausible', 'low_resolved_all')
STABILITY_THRESHOLD = 0.10
MID_NUMAX_RANGE = (0.15, 0.60)
HIGH_NUMAX_RANGE = (0.50, 1.80)
MID_HIGH_RANGE = (0.20, 0.50)
# If a reliable stellar mass column is present, optionally include log10(mass/Msun).
# Mass is not required for the default analysis; use it only with trusted metadata.
USE_MASS = False
MASS_COLUMNS = ('mass', 'mass_msun', 'Mass', 'MASS', 'mass_seismic', 'mass_apokasc')
# Also run the default no-mass specification when USE_MASS=True.
RUN_NO_MASS_BASELINE = True
# For direct V3/V4 comparison, always use paired stars and same covariates.
PAIRED_COMPONENTS = {
    'candidate_meso_frequency': ('long_component_nu_char_uHz', 'mid_component_nu_char_uHz'),
    'candidate_meso_rms': ('long_integrated_rms_native_sqrt_uHz', 'mid_integrated_rms_native_sqrt_uHz'),
    'candidate_gran_frequency': ('short_component_nu_char_uHz', 'high_component_nu_char_uHz'),
    'candidate_gran_rms': ('short_integrated_rms_native_sqrt_uHz', 'high_integrated_rms_native_sqrt_uHz'),
}
TARGETS = {
    'two_fixed': {
        'long_component_nu_char_uHz': 'Two-fixed long frequency (uHz)',
        'long_integrated_rms_native_sqrt_uHz': 'Two-fixed long RMS (native)',
        'short_component_nu_char_uHz': 'Two-fixed short frequency (uHz)',
        'short_integrated_rms_native_sqrt_uHz': 'Two-fixed short RMS (native)',
    },
    'three_fixed': {
        'low_component_nu_char_uHz': 'Three-fixed low frequency (uHz)',
        'low_integrated_rms_native_sqrt_uHz': 'Three-fixed low RMS (native)',
        'mid_component_nu_char_uHz': 'Three-fixed mid frequency (uHz)',
        'mid_integrated_rms_native_sqrt_uHz': 'Three-fixed mid RMS (native)',
        'high_component_nu_char_uHz': 'Three-fixed high frequency (uHz)',
        'high_integrated_rms_native_sqrt_uHz': 'Three-fixed high RMS (native)',
    },
}
# ==============================================================


def clean_numeric(series):
    s = pd.to_numeric(series, errors='coerce')
    return s.mask(~np.isfinite(s) | (s <= -999))


def bool_col(df, name, default=False):
    if name not in df:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[name].astype('string').str.strip().str.lower().isin(('true', '1', 'yes'))


def metadata_table(path):
    df = pd.read_csv(path, low_memory=False)
    needed = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate']
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f'Manifest missing columns: {sorted(missing)}')
    mass_col = next((c for c in MASS_COLUMNS if c in df.columns), None)
    if USE_MASS and mass_col is None:
        raise ValueError(f'USE_MASS=True but no mass column found among {MASS_COLUMNS}')
    cols = needed + ([mass_col] if mass_col else [])
    df = df[cols].copy()
    if mass_col:
        df = df.rename(columns={mass_col: 'stellar_mass'})
    else:
        df['stellar_mass'] = np.nan
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic']).copy()
    df['kic'] = df.kic.astype('int64')
    for c in ('nu_max', 'teff', 'metallicity', 'stellar_mass'):
        df[c] = clean_numeric(df[c])
    for c in ('nu_max', 'teff', 'stellar_mass'):
        df.loc[df[c] <= 0, c] = np.nan
    df['evolstate'] = df.evolstate.astype('string').str.strip().str.upper()
    for c in ('nu_max', 'teff', 'metallicity', 'stellar_mass', 'evolstate'):
        conflicts = int((df.groupby('kic')[c].nunique(dropna=True) > 1).sum())
        if conflicts:
            warnings.warn(f'{conflicts} KICs have conflicting {c} values across manifest rows')
    numeric = df.groupby('kic', as_index=False)[['nu_max', 'teff', 'metallicity', 'stellar_mass']].median()
    states = df.groupby('kic')['evolstate'].agg(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else pd.NA
    ).reset_index()
    return numeric.merge(states, on='kic', validate='one_to_one')


def read_model(path, model):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {model} results: {path}')
    df = pd.read_csv(path, low_memory=False)
    required = {'kic', 'duration', 'model', *TARGETS[model]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    df = df[df.model.astype(str).str.strip() == model].copy()
    if df.empty:
        raise ValueError(f'No {model} rows in {path}')
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic']).copy()
    df['kic'] = df.kic.astype('int64')
    df['duration'] = (df.duration.astype(str).str.strip().str.lower()
                      .str.replace('days', 'd', regex=False)
                      .str.replace(r'^(20|55|80)$', r'\1d', regex=True))
    df = df[df.duration.isin(FILTERS)].copy()
    if 'success' in df:
        df = df[bool_col(df, 'success')].copy()
    if df.duplicated(['kic', 'duration']).any():
        raise ValueError(f'{model}: duplicate KIC/filter rows after selection')
    numeric = list(TARGETS[model]) + ['oscillation_envelope_log_width', 'rms_log10_residual',
                                      'nu_max_catalogue']
    for c in numeric:
        if c in df:
            df[c] = clean_numeric(df[c])
    df['width_bound'] = False
    if 'oscillation_envelope_log_width' in df:
        w = df.oscillation_envelope_log_width
        df['width_bound'] = np.isclose(w, .08, atol=.01, rtol=0) | np.isclose(w, .8, atol=.01, rtol=0)
    # V4 fixed-alpha models cannot have alpha-bound hits. Include any supplied char-bound flags.
    char_flags = [c for c in df if c.endswith('_char_near_parameter_bound')]
    df['char_bound'] = False
    for c in char_flags:
        df['char_bound'] |= bool_col(df, c)
    df['any_bound'] = df.width_bound | df.char_bound
    df['model'] = model
    return df


def add_cross_filter_flags(three):
    """One per-star quality table, derived only from three_fixed fits."""
    rows = []
    for kic, g in three.groupby('kic', sort=False):
        d = {'kic': int(kic), 'n_filters': int(g.duration.nunique())}
        complete = set(FILTERS).issubset(set(g.duration))
        d['all_filters'] = complete
        for kind, col in [('freq', 'mid_component_nu_char_uHz'),
                          ('rms', 'mid_integrated_rms_native_sqrt_uHz')]:
            vals = pd.to_numeric(g[col], errors='coerce').to_numpy(float)
            good = complete and len(vals) == len(FILTERS) and np.all(np.isfinite(vals) & (vals > 0))
            d[f'mid_{kind}_spread'] = float(vals.max() / vals.min() - 1) if good else np.nan
        d['mid_stable_both_10pct'] = bool(
            np.isfinite(d['mid_freq_spread']) and np.isfinite(d['mid_rms_spread'])
            and d['mid_freq_spread'] <= STABILITY_THRESHOLD
            and d['mid_rms_spread'] <= STABILITY_THRESHOLD)
        low_below = bool_col(g, 'low_char_below_fit_band')
        low_below_filter = bool_col(g, 'low_char_below_filter_timescale')
        d['low_below_band_any'] = bool(low_below.any())
        d['low_below_filter_any'] = bool(low_below_filter.any())
        d['low_resolved_all'] = bool(complete and not low_below.any() and not low_below_filter.any())
        # Physical plausibility must hold for ALL filters; do not use the median to conceal failures.
        ratios = []
        for _, r in g.iterrows():
            nu = r.get('nu_max_catalogue', np.nan)
            mid = r.get('mid_component_nu_char_uHz', np.nan)
            high = r.get('high_component_nu_char_uHz', np.nan)
            if not all(np.isfinite(v) and v > 0 for v in (nu, mid, high)):
                ratios.append(False)
            else:
                ratios.append(MID_NUMAX_RANGE[0] <= mid / nu <= MID_NUMAX_RANGE[1]
                              and HIGH_NUMAX_RANGE[0] <= high / nu <= HIGH_NUMAX_RANGE[1]
                              and MID_HIGH_RANGE[0] <= mid / high <= MID_HIGH_RANGE[1])
        d['plausible_all_filters'] = bool(complete and len(ratios) == len(FILTERS) and all(ratios))
        d['stable_mid_plausible'] = d['mid_stable_both_10pct'] and d['plausible_all_filters']
        rows.append(d)
    return pd.DataFrame(rows)


def sample_mask(df, sample):
    if sample == 'all_valid':
        return pd.Series(True, index=df.index)
    if sample == 'no_bounds':
        return ~df.any_bound
    if sample == 'stable_mid_10pct':
        return df.mid_stable_both_10pct.fillna(False).astype(bool)
    if sample == 'stable_mid_plausible':
        return df.stable_mid_plausible.fillna(False).astype(bool)
    if sample == 'low_resolved_all':
        return df.low_resolved_all.fillna(False).astype(bool)
    raise ValueError(f'Unknown sample {sample}')


def design(df, target, with_mass=False):
    cols = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate', target]
    if with_mass:
        cols.append('stellar_mass')
    d = df[cols].dropna().copy()
    valid = (d.nu_max > 0) & (d.teff > 0) & (d[target] > 0) & d.evolstate.isin(['RGB', 'RC'])
    if with_mass:
        valid &= d.stellar_mass > 0
    d = d[valid].copy()
    d['log_nu'] = np.log10(d.nu_max / 40.)
    d['log_teff'] = np.log10(d.teff / 4800.)
    d['is_rc'] = (d.evolstate == 'RC').astype(float)
    d['log_y'] = np.log10(d[target])
    if with_mass:
        d['log_mass'] = np.log10(d.stellar_mass)
    return d


def matrix(d, metallicity=True, with_mass=False):
    names = ['intercept', 'log10(nu_max/40uHz)', 'log10(teff/4800K)', 'RC_indicator']
    columns = [np.ones(len(d)), d.log_nu.to_numpy(), d.log_teff.to_numpy(), d.is_rc.to_numpy()]
    if with_mass:
        columns.append(d.log_mass.to_numpy())
        names.append('log10(mass/Msun)')
    if metallicity:
        columns.append(d.metallicity.to_numpy())
        names.append('[Fe/H]')
    return np.column_stack(columns).astype(float), names


def fit_ols(d, metallicity=True, with_mass=False):
    X, names = matrix(d, metallicity, with_mass)
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
    return {'n': int(n), 'r_squared': r2, 'coefficients': coefficients}, resid


def regress(df, target, with_mass=False):
    t = design(df, target, with_mass)
    baseline, resid = fit_ols(t, metallicity=False, with_mass=with_mass)
    extended, _ = fit_ols(t, metallicity=True, with_mass=with_mass)
    rho = spearmanr(t.metallicity.to_numpy(), resid) if len(t) >= 4 else None
    return t, {'baseline': baseline, 'with_metallicity': extended,
               'delta_r_squared': extended['r_squared'] - baseline['r_squared'],
               'baseline_residual_spearman': {
                   'rho': float(rho.statistic), 'p': float(rho.pvalue)} if rho else None}


def plot_target(df, target, model, duration, out):
    t = design(df, target)
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for state, color in [('RGB', 'tab:blue'), ('RC', 'tab:orange')]:
        s = t[t.evolstate == state]
        axes[0].scatter(s.nu_max, s[target], s=9, alpha=.35, color=color,
                        label=f'{state} ({len(s)})', rasterized=True)
        axes[1].scatter(s.metallicity, s.log_y, s=9, alpha=.35, color=color,
                        label=state, rasterized=True)
    axes[0].set(xscale='log', yscale='log', xlabel=r'Catalogue $\nu_{\max}$ ($\mu$Hz)',
                ylabel=TARGETS[model].get(target, target))
    axes[1].set(xlabel='[Fe/H] (dex)', ylabel=f'log10({target})')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(f'{model} | {duration}')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def paired_difference(a, b, target_a, target_b, rng, with_mass=False):
    """Paired stellar bootstrap: beta_three_fixed - beta_two_fixed.

    The two models use identical stars and design matrices. We regress the
    per-star difference in log target on the shared covariates for each draw.
    """
    da = design(a, target_a, with_mass)
    db = design(b, target_b, with_mass)
    common = da.merge(db, on='kic', suffixes=('_two', '_three'), validate='one_to_one')
    if common.empty:
        raise ValueError('No matched stars')
    covars = ['log_nu', 'log_teff', 'is_rc', 'metallicity'] + (['log_mass'] if with_mass else [])
    for col in covars:
        if not np.allclose(common[f'{col}_two'], common[f'{col}_three'], rtol=0, atol=1e-10):
            raise ValueError(f'Conflicting paired covariates: {col}')
    X = np.column_stack([np.ones(len(common)), common.log_nu_two, common.log_teff_two,
                         common.is_rc_two] + ([common.log_mass_two] if with_mass else [])
                        + [common.metallicity_two])
    y0 = common.log_y_two.to_numpy(dtype=float)
    y1 = common.log_y_three.to_numpy(dtype=float)
    if len(common) < MIN_REGRESSION_N or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError('Too few matched stars or rank-deficient design')
    b0 = float(np.linalg.lstsq(X, y0, rcond=None)[0][-1])
    b1 = float(np.linalg.lstsq(X, y1, rcond=None)[0][-1])
    diffs = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(common), len(common))
        xx = X[idx]
        if np.linalg.matrix_rank(xx) < xx.shape[1]:
            continue
        diffs.append(float(np.linalg.lstsq(xx, (y1 - y0)[idx], rcond=None)[0][-1]))
    if not diffs:
        raise ValueError('No valid bootstrap resamples')
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'n_matched': int(len(common)), 'feh_beta_two_fixed': b0,
            'feh_beta_three_fixed': b1, 'difference_three_minus_two': b1 - b0,
            'bootstrap_ci95_lower': float(lo), 'bootstrap_ci95_upper': float(hi),
            'n_bootstrap_valid': len(diffs)}


def main():
    if ANALYSIS not in FIT_FILES:
        raise ValueError(f'Unknown ANALYSIS={ANALYSIS}')
    if PRIMARY_FILTER not in FILTERS:
        raise ValueError('Invalid PRIMARY_FILTER')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = metadata_table(MANIFEST)
    raw = {model: read_model(path, model) for model, path in FIT_FILES[ANALYSIS].items()}
    if 'three_fixed' not in raw:
        raise ValueError('three_fixed is required for cross-filter quality flags')
    quality = add_cross_filter_flags(raw['three_fixed'])
    quality.to_csv(OUTPUT_DIR / 'three_fixed_cross_filter_quality.csv', index=False)
    models = {}
    counts = {}
    for model, fits in raw.items():
        merged = fits.merge(meta, on='kic', how='left', validate='many_to_one', indicator=True)
        merged = merged.merge(quality, on='kic', how='left', validate='many_to_one')
        merged['valid_stellar_metadata'] = (merged.nu_max.gt(0) & merged.teff.gt(0)
                                            & merged.metallicity.notna()
                                            & merged.evolstate.isin(['RGB', 'RC']))
        merged.to_csv(OUTPUT_DIR / f'{model}_all_filters_merged.csv', index=False)
        models[model] = merged
        counts[model] = {'fits': len(fits), 'unique_kics': fits.kic.nunique(),
                         'unmatched_metadata_rows': int((merged._merge == 'left_only').sum())}
        print(f'{model}: {len(fits)} fits, {fits.kic.nunique()} KICs; '
              f'{counts[model]["unmatched_metadata_rows"]} unmatched metadata rows')
    specs = [False] + ([True] if USE_MASS else [])
    if USE_MASS and not RUN_NO_MASS_BASELINE:
        specs = [True]
    summary = []
    sample_counts = []
    for model, merged in models.items():
        for duration in FILTERS:
            d = merged[(merged.duration == duration) & merged.valid_stellar_metadata].copy()
            for sample in SAMPLES:
                sub = d[sample_mask(d, sample)].copy()
                sample_counts.append({'model': model, 'duration': duration, 'sample': sample,
                                      'n': len(sub)})
                sub.to_csv(OUTPUT_DIR / f'{model}_{duration}_{sample}_stars.csv', index=False)
                for target in TARGETS[model]:
                    for with_mass in specs:
                        spec = 'with_mass' if with_mass else 'no_mass'
                        try:
                            t, result = regress(sub, target, with_mass)
                            c = result['with_metallicity']['coefficients']['[Fe/H]']
                            summary.append({'model': model, 'duration': duration, 'sample': sample,
                                            'specification': spec, 'target': target, 'n': len(t),
                                            'feh_coefficient': c['estimate'], 'feh_hc3_se': c['hc3_se'],
                                            'feh_ci95_lower': c['ci95_lower'],
                                            'feh_ci95_upper': c['ci95_upper'],
                                            'feh_p_value': c['p_value_normal_approx'],
                                            'baseline_r2': result['baseline']['r_squared'],
                                            'extended_r2': result['with_metallicity']['r_squared'],
                                            'delta_r2': result['delta_r_squared'],
                                            'baseline_residual_spearman_rho': result['baseline_residual_spearman']['rho'],
                                            'error': ''})
                            if MAKE_PLOTS and sample == 'all_valid' and not with_mass:
                                plot_target(sub, target, model, duration,
                                            OUTPUT_DIR / f'{model}_{duration}_{target}.png')
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            summary.append({'model': model, 'duration': duration, 'sample': sample,
                                            'specification': spec, 'target': target,
                                            'n': len(design(sub, target, with_mass)), 'error': str(exc)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'regression_summary.csv', index=False)
    paired_rows = []
    if 'two_fixed' in models and 'three_fixed' in models:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for duration in FILTERS:
            for sample in SAMPLES:
                a = models['two_fixed']
                b = models['three_fixed']
                a = a[(a.duration == duration) & a.valid_stellar_metadata].copy()
                b = b[(b.duration == duration) & b.valid_stellar_metadata].copy()
                a = a[sample_mask(a, sample)].copy()
                b = b[sample_mask(b, sample)].copy()
                for pair_name, (target_a, target_b) in PAIRED_COMPONENTS.items():
                    for with_mass in specs:
                        spec = 'with_mass' if with_mass else 'no_mass'
                        try:
                            res = paired_difference(a, b, target_a, target_b, rng, with_mass)
                            paired_rows.append({'duration': duration, 'sample': sample,
                                                'specification': spec, 'comparison': pair_name,
                                                'two_target': target_a, 'three_target': target_b,
                                                **res, 'error': ''})
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            paired_rows.append({'duration': duration, 'sample': sample,
                                                'specification': spec, 'comparison': pair_name,
                                                'two_target': target_a, 'three_target': target_b,
                                                'error': str(exc)})
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUTPUT_DIR / 'paired_model_comparison.csv', index=False)
    counts_df = pd.DataFrame(sample_counts)
    counts_df.to_csv(OUTPUT_DIR / 'sample_counts.csv', index=False)
    config = {'analysis': ANALYSIS, 'manifest': str(MANIFEST),
              'fit_files': {k: str(v) for k, v in FIT_FILES[ANALYSIS].items()},
              'primary_filter': PRIMARY_FILTER, 'bootstrap_reps': BOOTSTRAP_REPS,
              'bootstrap_seed': BOOTSTRAP_SEED, 'counts': counts,
              'quality_flags': {'stability_threshold': STABILITY_THRESHOLD,
                                'mid_numax_range': MID_NUMAX_RANGE,
                                'high_numax_range': HIGH_NUMAX_RANGE,
                                'mid_high_range': MID_HIGH_RANGE},
              'use_mass': USE_MASS,
              'caveats': [
                  '20d/55d/80d are correlated high-pass filtered versions of the same Kepler observations.',
                  'HC3 SE are heteroskedasticity-robust; they do not model covariate measurement errors.',
                  'Bootstrap resamples unique KICs within each filter, not spectra across filters.',
                  'Native RMS amplitude units are not assumed to be ppm.',
                  'Frequency-sorted components do not have confirmed physical identities.',
                  'Quality cuts can introduce selection bias, especially if correlated with metallicity.',
                  'Mass may correlate with seismic covariates; check model rank and measurement provenance.',
                  'The 300-star pilot may not be representative of the full sample.',
                  'Exploratory associations, not causal estimates.']}
    (OUTPUT_DIR / 'analysis_summary.json').write_text(json.dumps(config, indent=2, default=str))
    lines = [f'V3/V4 GRANULATION SCALING: {ANALYSIS}',
             f'Primary filter: {PRIMARY_FILTER}',
             'HC3 robust OLS; matched-star bootstrap of three_fixed minus two_fixed [Fe/H] slope.',
             'The three filters are correlated reductions of the same ~4-year observations.',
             'Physical identity of the frequency-sorted components is not guaranteed.', '']
    cols = ['model', 'sample', 'specification', 'target', 'n', 'feh_coefficient',
            'feh_hc3_se', 'feh_ci95_lower', 'feh_ci95_upper', 'delta_r2', 'error']
    show = summary_df[summary_df.duration == PRIMARY_FILTER]
    lines += ['INDIVIDUAL REGRESSIONS (primary filter):',
              show.reindex(columns=cols).to_string(index=False, float_format=lambda x: f'{x:.4g}')]
    if not paired_df.empty:
        showp = paired_df[paired_df.duration == PRIMARY_FILTER]
        cols_p = ['sample', 'specification', 'comparison', 'n_matched', 'feh_beta_two_fixed',
                  'feh_beta_three_fixed', 'difference_three_minus_two',
                  'bootstrap_ci95_lower', 'bootstrap_ci95_upper', 'error']
        lines += ['', 'PAIRED MODEL DIFFERENCES (three_fixed minus two_fixed):',
                  showp.reindex(columns=cols_p).to_string(index=False, float_format=lambda x: f'{x:.4g}')]
    lines += ['', 'SAMPLE COUNTS (primary filter):',
              counts_df[counts_df.duration == PRIMARY_FILTER].to_string(index=False)]
    (OUTPUT_DIR / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved to: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
