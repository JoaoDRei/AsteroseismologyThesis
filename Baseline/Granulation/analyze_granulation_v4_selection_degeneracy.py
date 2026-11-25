#!/usr/bin/env python3
"""V4 pilot: selection bias, component trade-offs and metallicity of ratios.

Run: python -m Baseline.analysis.analyze_granulation_v4_selection_degeneracy
Edit CONFIG only. No fitting is performed. 20d/55d/80d are correlated KADACS
high-pass reductions of the same Kepler observations, not independent stars.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm, mannwhitneyu

# ============================ CONFIG ============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIT_CSV = PROJECT_ROOT / 'data/fitted_parameters_v4_three_fixed_all/fit_results.csv'
MANIFEST = PROJECT_ROOT / 'Baseline/manifest.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/granulation_v4_selection_degeneracy_full'
FILTERS = ('20d', '55d', '80d')
PRIMARY_FILTER = '55d'
STABILITY_TOL = 0.10
MID_NUMAX_RANGE = (0.15, 0.60)
HIGH_NUMAX_RANGE = (0.50, 1.80)
MID_HIGH_RANGE = (0.20, 0.50)
BOOTSTRAP_REPS = 1000
SEED = 42
MIN_REGRESSION_N = 30
MASS_COLUMNS = ('mass', 'mass_msun', 'Mass', 'MASS', 'mass_seismic', 'mass_apokasc')
RUN_MASS_SENSITIVITY = False  # requires an appropriate mass column in manifest
MAKE_PLOTS = True
# ================================================================

FREQ = {k: f'{k}_component_nu_char_uHz' for k in ('low', 'mid', 'high')}
AMP = {k: f'{k}_integrated_rms_native_sqrt_uHz' for k in ('low', 'mid', 'high')}
RATIO_COLS = ('mid_over_high_frequency', 'mid_over_high_rms',
              'mid_over_numax', 'high_over_numax')
SAMPLES = ('all_valid', 'stable_mid', 'unstable_mid', 'low_resolved_all',
           'stable_and_low_resolved', 'stable_plausible')


def numeric(x):
    y = pd.to_numeric(x, errors='coerce')
    return y.mask(~np.isfinite(y) | (y <= -999))


def bool_values(x):
    return x.astype('string').str.strip().str.lower().isin(('true', '1', 'yes'))


def safe_ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan),
                     where=np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0))


def read_metadata():
    raw = pd.read_csv(MANIFEST, low_memory=False)
    required = ('kic', 'nu_max', 'teff', 'metallicity', 'evolstate')
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Manifest missing columns: {sorted(missing)}')
    mass = next((c for c in MASS_COLUMNS if c in raw.columns), None)
    if RUN_MASS_SENSITIVITY and mass is None:
        raise ValueError('RUN_MASS_SENSITIVITY=True but no mass column found')
    raw = raw[list(required) + ([mass] if mass else [])].copy()
    raw['stellar_mass'] = numeric(raw[mass]) if mass else np.nan
    raw['kic'] = pd.to_numeric(raw.kic, errors='coerce')
    raw = raw.dropna(subset=['kic']).copy()
    raw['kic'] = raw.kic.astype('int64')
    for col in ('nu_max', 'teff', 'metallicity', 'stellar_mass'):
        raw[col] = numeric(raw[col])
    for col in ('nu_max', 'teff', 'stellar_mass'):
        raw.loc[raw[col] <= 0, col] = np.nan
    raw['evolstate'] = raw.evolstate.astype('string').str.strip().str.upper()
    for col in ('nu_max', 'teff', 'metallicity', 'stellar_mass', 'evolstate'):
        n = int((raw.groupby('kic')[col].nunique(dropna=True) > 1).sum())
        if n:
            warnings.warn(f'{n} stars have conflicting manifest {col} values; using median/mode')
    med = raw.groupby('kic', as_index=False)[['nu_max', 'teff', 'metallicity', 'stellar_mass']].median()
    state = raw.groupby('kic').evolstate.agg(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else pd.NA
    ).reset_index()
    return med.merge(state, on='kic', validate='one_to_one')


def read_fits():
    df = pd.read_csv(FIT_CSV, low_memory=False)
    needed = {'kic', 'duration', 'model', *FREQ.values(), *AMP.values()}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Fit CSV missing columns: {sorted(missing)}')
    df = df[df.model.astype(str).str.strip().eq('three_fixed')].copy()
    if 'success' in df:
        df = df[bool_values(df.success)].copy()
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic']).copy()
    df['kic'] = df.kic.astype('int64')
    df['duration'] = (df.duration.astype(str).str.strip().str.lower()
                      .str.replace('days', 'd', regex=False)
                      .str.replace(r'^(20|55|80)$', r'\1d', regex=True))
    df = df[df.duration.isin(FILTERS)].copy()
    if df.empty or df.duplicated(['kic', 'duration']).any():
        raise ValueError('No three_fixed fits or duplicate KIC/filter rows')
    for col in [*FREQ.values(), *AMP.values(), 'rms_log10_residual']:
        if col in df:
            df[col] = numeric(df[col])
    df['mid_over_high_frequency'] = safe_ratio(df[FREQ['mid']], df[FREQ['high']])
    df['mid_over_high_rms'] = safe_ratio(df[AMP['mid']], df[AMP['high']])
    # Catalogue nu_max from the manifest is merged below, not inferred from fitted envelope.
    for flag in ('low_char_below_fit_band', 'low_char_below_filter_timescale'):
        df[flag] = bool_values(df[flag]) if flag in df else False
    return df


def make_star_quality(df):
    records = []
    for kic, group in df.groupby('kic'):
        complete = len(group) == len(FILTERS) and set(group.duration) == set(FILTERS)
        d = {'kic': int(kic), 'n_filters': len(group), 'all_filters': complete}
        for component in ('mid', 'high'):
            for metric, columns in [('freq', FREQ), ('rms', AMP)]:
                vals = group[columns[component]].to_numpy(float)
                good = complete and np.all(np.isfinite(vals) & (vals > 0))
                d[f'{component}_{metric}_spread'] = float(vals.max()/vals.min()-1) if good else np.nan
        d['stable_mid'] = bool(complete and d['mid_freq_spread'] <= STABILITY_TOL
                               and d['mid_rms_spread'] <= STABILITY_TOL)
        d['low_below_band_any'] = bool(group.low_char_below_fit_band.any())
        d['low_below_filter_any'] = bool(group.low_char_below_filter_timescale.any())
        d['low_resolved_all'] = bool(complete and not d['low_below_band_any']
                                     and not d['low_below_filter_any'])
        valid = (group.mid_over_numax.between(*MID_NUMAX_RANGE) &
                 group.high_over_numax.between(*HIGH_NUMAX_RANGE) &
                 group.mid_over_high_frequency.between(*MID_HIGH_RANGE))
        d['plausible_all'] = bool(complete and valid.all())
        d['stable_plausible'] = d['stable_mid'] and d['plausible_all']
        for ratio in RATIO_COLS:
            vals = group[ratio].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            d[f'{ratio}_star_median'] = float(np.median(vals)) if len(vals) else np.nan
            d[f'{ratio}_spread'] = float(vals.max()/vals.min()-1) if complete and len(vals)==3 else np.nan
        records.append(d)
    return pd.DataFrame(records)


def sample_select(df, sample):
    if sample == 'all_valid': return np.ones(len(df), dtype=bool)
    if sample == 'stable_mid': return df.stable_mid.fillna(False).to_numpy(bool)
    if sample == 'unstable_mid': return (~df.stable_mid.fillna(False)).to_numpy(bool)
    if sample == 'low_resolved_all': return df.low_resolved_all.fillna(False).to_numpy(bool)
    if sample == 'stable_and_low_resolved':
        return (df.stable_mid.fillna(False) & df.low_resolved_all.fillna(False)).to_numpy(bool)
    if sample == 'stable_plausible': return df.stable_plausible.fillna(False).to_numpy(bool)
    raise ValueError(sample)


def regression(df, target, with_mass=False):
    cols = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate', target]
    if with_mass: cols.append('stellar_mass')
    d = df[cols].dropna().copy()
    d = d[(d.nu_max > 0) & (d.teff > 0) & (d[target] > 0) &
          d.evolstate.isin(['RGB', 'RC'])]
    if with_mass: d = d[d.stellar_mass > 0]
    n = len(d)
    if n < MIN_REGRESSION_N:
        raise ValueError(f'Only {n} valid stars')
    y = np.log10(d[target].to_numpy(float))
    columns = [np.ones(n), np.log10(d.nu_max.to_numpy(float)/40),
               np.log10(d.teff.to_numpy(float)/4800),
               (d.evolstate == 'RC').to_numpy(float)]
    names = ['intercept', 'log_nu', 'log_teff', 'is_rc']
    if with_mass:
        columns.append(np.log10(d.stellar_mass.to_numpy(float)))
        names.append('log_mass')
    X0 = np.column_stack(columns)
    X1 = np.column_stack([X0, d.metallicity.to_numpy(float)])
    if np.linalg.matrix_rank(X1) < X1.shape[1]:
        raise ValueError('Rank-deficient regression')

    def ols(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y-X@beta
        inverse = np.linalg.pinv(X.T@X)
        leverage = np.einsum('ij,jk,ik->i', X, inverse, X)
        w = (residual/np.maximum(1-leverage, 1e-8))**2
        cov = inverse @ (X.T @ (X*w[:,None])) @ inverse
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        sst = np.sum((y-y.mean())**2)
        r2 = 1-np.sum(residual**2)/sst if sst>0 else np.nan
        return beta, se, float(r2), residual

    _, _, base_r2, base_resid = ols(X0)
    beta, se, full_r2, _ = ols(X1)
    b, s = float(beta[-1]), float(se[-1])
    rho = spearmanr(d.metallicity, base_resid)
    return {'n': n, 'beta_feh': b, 'hc3_se': s, 'ci95_low': b-1.96*s,
            'ci95_high': b+1.96*s, 'p_normal': float(2*norm.sf(abs(b/s))) if s>0 else np.nan,
            'baseline_r2': base_r2, 'full_r2': full_r2, 'delta_r2': full_r2-base_r2,
            'baseline_residual_spearman': float(rho.statistic)}


def describe_groups(stars):
    rows=[]
    for group, g in [('stable', stars[stars.stable_mid]),
                     ('unstable', stars[~stars.stable_mid])]:
        r={'group':group,'n':len(g), 'rgb_fraction':float(g.evolstate.eq('RGB').mean()),
           'rc_fraction':float(g.evolstate.eq('RC').mean()),
           'low_resolved_fraction':float(g.low_resolved_all.mean())}
        for col in ('metallicity','nu_max','teff','stellar_mass'):
            a=numeric(g[col]).dropna()
            r[f'{col}_n']=len(a)
            r[f'{col}_median']=float(a.median()) if len(a) else np.nan
            r[f'{col}_p16']=float(a.quantile(.16)) if len(a) else np.nan
            r[f'{col}_p84']=float(a.quantile(.84)) if len(a) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def group_tests(stars):
    a=stars[stars.stable_mid]
    b=stars[~stars.stable_mid]
    rows=[]
    for col in ('metallicity','nu_max','teff','stellar_mass'):
        x=numeric(a[col]).dropna().to_numpy(float)
        y=numeric(b[col]).dropna().to_numpy(float)
        p=float(mannwhitneyu(x,y,alternative='two-sided').pvalue) if len(x)>0 and len(y)>0 else np.nan
        rows.append({'variable':col,'stable_n':len(x),'unstable_n':len(y),
                     'median_stable':float(np.median(x)) if len(x) else np.nan,
                     'median_unstable':float(np.median(y)) if len(y) else np.nan,
                     'mann_whitney_p_exploratory':p})
    return pd.DataFrame(rows)


def tradeoff_analysis(df):
    # Correlate filter-to-filter deviations within each star, so between-star
    # astrophysical scaling cannot create spurious within-star compensation.
    rows=[]
    for kic, g in df.groupby('kic'):
        if len(g)!=3 or set(g.duration)!=set(FILTERS): continue
        record={'kic':int(kic)}
        for key, col in [('mid_rms', AMP['mid']),('high_rms', AMP['high']),
                         ('mid_freq', FREQ['mid']),('high_freq', FREQ['high'])]:
            vals=g[col].to_numpy(float)
            record[key+'_spread']=float(vals.max()/vals.min()-1) if np.all(np.isfinite(vals)&(vals>0)) else np.nan
        rows.append(record)
    spreads=pd.DataFrame(rows)
    deviations=[]
    for kic,g in df.groupby('kic'):
        if len(g)!=3 or set(g.duration)!=set(FILTERS):continue
        r={'kic':int(kic)}
        for metric, mapping in [('rms',AMP),('freq',FREQ)]:
            for component in ('mid','high'):
                values=g[mapping[component]].to_numpy(float)
                if not np.all(np.isfinite(values)&(values>0)):
                    r[f'{component}_{metric}']=None
                else:
                    r[f'{component}_{metric}']=np.log10(values)-np.log10(values).mean()
        for i,(_,v) in enumerate(g.iterrows()):
            deviations.append({'kic':int(kic),'duration':v.duration,
                **{k:(z[i] if z is not None else np.nan) for k,z in r.items() if k!='kic'}})
    deviations=pd.DataFrame(deviations)
    stats=[]
    for metric in ('rms','freq'):
        x=deviations[f'mid_{metric}'].to_numpy(float)
        y=deviations[f'high_{metric}'].to_numpy(float)
        valid=np.isfinite(x)&np.isfinite(y)
        rho=spearmanr(x[valid],y[valid]) if valid.sum()>=4 else None
        stats.append({'metric':metric,'n_filter_rows':int(valid.sum()),
                      'n_unique_stars':int(deviations.loc[valid,'kic'].nunique()),
                      'within_star_deviation_spearman':float(rho.statistic) if rho else np.nan,
                      'note':'Descriptive only: filter rows are correlated; no naive p-value.'})
    return spreads, deviations, pd.DataFrame(stats)


def paired_bootstrap_slope_difference(stars, target, sample_a, sample_b, rng):
    # Independent groups (stable vs unstable); bootstrap unique stars in each.
    a=stars[sample_select(stars,sample_a)].copy()
    b=stars[sample_select(stars,sample_b)].copy()
    def slope(d):
        return regression(d,target)['beta_feh']
    observed=slope(a)-slope(b)
    vals=[]
    for _ in range(BOOTSTRAP_REPS):
        aa=a.iloc[rng.integers(0,len(a),len(a))]
        bb=b.iloc[rng.integers(0,len(b),len(b))]
        try: vals.append(slope(aa)-slope(bb))
        except (ValueError,np.linalg.LinAlgError): continue
    if not vals: raise ValueError('No valid bootstrap resamples')
    return {'target':target,'comparison':f'{sample_a} minus {sample_b}',
            'n_a':len(a),'n_b':len(b),'difference':observed,
            'ci95_low':float(np.percentile(vals,2.5)),
            'ci95_high':float(np.percentile(vals,97.5)),
            'bootstrap_valid':len(vals)}


def plot_diagnostics(stars, primary, deviations):
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for i,col in enumerate(('metallicity','nu_max','teff','stellar_mass')):
        a=ax.flat[i]
        for mask,label in ((stars.stable_mid,'stable'),(~stars.stable_mid,'unstable')):
            v=numeric(stars.loc[mask,col]).dropna()
            if len(v): a.hist(v,bins=24,alpha=.5,density=True,label=f'{label} (n={len(v)})')
        a.set_xlabel(col);a.set_ylabel('Density')
        if a.has_data(): a.legend(fontsize=8)
        if col=='nu_max':a.set_xscale('log')
    fig.suptitle('Selection: stable vs unstable middle component')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'selection_distributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for i,ratio in enumerate(('mid_over_high_frequency','mid_over_high_rms')):
        for state in ('RGB','RC'):
            g=primary[primary.evolstate.eq(state)]
            ax[i].scatter(g.metallicity,g[ratio],s=12,alpha=.45,label=state,rasterized=True)
        ax[i].set(xlabel='[Fe/H] (dex)',ylabel=ratio)
        ax[i].legend();ax[i].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'ratios_vs_metallicity.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for i,metric in enumerate(('rms','freq')):
        x=deviations[f'mid_{metric}'];y=deviations[f'high_{metric}']
        ax[i].scatter(x,y,s=12,alpha=.4,rasterized=True)
        ax[i].axhline(0,color='gray',lw=.7);ax[i].axvline(0,color='gray',lw=.7)
        ax[i].set(xlabel=f'Mid {metric} log10 deviation from star mean',
                  ylabel=f'High {metric} log10 deviation from star mean')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'within_star_component_tradeoffs.png',dpi=160);plt.close(fig)


def main():
    if PRIMARY_FILTER not in FILTERS:raise ValueError('Invalid PRIMARY_FILTER')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    fits=read_fits()
    meta=read_metadata()
    fits=fits.merge(meta,on='kic',how='left',validate='many_to_one',indicator=True)
    print(f'Fits: {len(fits)}; stars: {fits.kic.nunique()}; unmatched metadata rows: {(fits._merge=="left_only").sum()}')
    fits['mid_over_numax']=safe_ratio(fits[FREQ['mid']],fits.nu_max)
    fits['high_over_numax']=safe_ratio(fits[FREQ['high']],fits.nu_max)
    quality=make_star_quality(fits)
    stars=quality.merge(meta,on='kic',how='left',validate='one_to_one')
    fits=fits.merge(quality,on='kic',how='left',validate='many_to_one')
    primary=fits[fits.duration==PRIMARY_FILTER].copy()
    if primary.empty:raise ValueError(f'No {PRIMARY_FILTER} rows')
    stars=stars.merge(primary[['kic',*FREQ.values(),*AMP.values(),*RATIO_COLS]],
                      on='kic',how='left',validate='one_to_one')
    fits.to_csv(OUTPUT_DIR/'per_filter_merged.csv',index=False)
    stars.to_csv(OUTPUT_DIR/'per_star_quality_and_ratios.csv',index=False)
    groups=describe_groups(stars)
    tests=group_tests(stars)
    groups.to_csv(OUTPUT_DIR/'stable_unstable_descriptives.csv',index=False)
    tests.to_csv(OUTPUT_DIR/'stable_unstable_tests.csv',index=False)
    spreads,deviations,tradeoffs=tradeoff_analysis(fits)
    spreads.to_csv(OUTPUT_DIR/'per_star_component_spreads.csv',index=False)
    deviations.to_csv(OUTPUT_DIR/'within_star_filter_deviations.csv',index=False)
    tradeoffs.to_csv(OUTPUT_DIR/'within_star_tradeoff_summary.csv',index=False)
    reg=[]
    targets=list(RATIO_COLS)+[AMP['mid'],AMP['high'],FREQ['mid'],FREQ['high']]
    for duration in FILTERS:
        subset=fits[fits.duration==duration]
        for sample in SAMPLES:
            data=subset[sample_select(subset,sample)]
            for target in targets:
                for with_mass in ([False,True] if RUN_MASS_SENSITIVITY else [False]):
                    row={'duration':duration,'sample':sample,'target':target,
                         'specification':'with_mass' if with_mass else 'no_mass'}
                    try:row.update(regression(data,target,with_mass=with_mass));row['error']=''
                    except (ValueError,np.linalg.LinAlgError) as e:row['error']=str(e)
                    reg.append(row)
    reg=pd.DataFrame(reg)
    reg.to_csv(OUTPUT_DIR/'regression_summary.csv',index=False)
    # Per-star median ratios provide a one-observation-per-star cross-filter check.
    for target in RATIO_COLS:
        stars[target+'_per_star_median']=stars[target+'_star_median']
    star_reg=[]
    for sample in SAMPLES:
        data=stars[sample_select(stars,sample)]
        for target in [x+'_per_star_median' for x in RATIO_COLS]:
            row={'sample':sample,'target':target}
            try:row.update(regression(data,target));row['error']=''
            except (ValueError,np.linalg.LinAlgError) as e:row['error']=str(e)
            star_reg.append(row)
    pd.DataFrame(star_reg).to_csv(OUTPUT_DIR/'per_star_median_ratio_regressions.csv',index=False)
    rng=np.random.default_rng(SEED)
    differences=[]
    for target in ('mid_over_high_frequency','mid_over_high_rms',AMP['mid'],AMP['high']):
        try:differences.append(paired_bootstrap_slope_difference(
            primary,target,'stable_mid','unstable_mid',rng))
        except (ValueError,np.linalg.LinAlgError) as e:
            differences.append({'target':target,'error':str(e)})
    pd.DataFrame(differences).to_csv(OUTPUT_DIR/'stable_vs_unstable_slope_bootstrap.csv',index=False)
    if MAKE_PLOTS:plot_diagnostics(stars,primary,deviations)
    report=['V4 SELECTION, DEGENERACY AND METALLICITY-RATIO DIAGNOSTICS',
            f'Input: {FIT_CSV}',f'Unique stars: {len(stars)}; primary filter: {PRIMARY_FILTER}',
            'KADACS filters are correlated reductions of the same observations.',
            'Ratios and component labels do not establish physical identities.',
            'Stability selection is based on fitted parameters and can induce selection bias.',
            'All tests are exploratory; multiple comparisons are not corrected.','',
            'STABLE VS UNSTABLE SAMPLE CHARACTERISTICS:',groups.to_string(index=False,float_format=lambda x:f'{x:.4g}'),
            '', 'COVARIATE DIFFERENCES (descriptive unadjusted tests):',
            tests.to_string(index=False,float_format=lambda x:f'{x:.4g}'),'',
            'WITHIN-STAR COMPONENT DEVIATION CORRELATIONS:',
            tradeoffs.to_string(index=False,float_format=lambda x:f'{x:.4g}'),'',
            'PRIMARY FILTER METALLICITY REGRESSIONS:',
            reg[(reg.duration==PRIMARY_FILTER)&(reg.specification=='no_mass')]
            [['sample','target','n','beta_feh','hc3_se','ci95_low','ci95_high','delta_r2','error']]
            .to_string(index=False,float_format=lambda x:f'{x:.4g}'),'',
            'STABLE MINUS UNSTABLE METALLICITY SLOPES (star bootstrap):',
            pd.DataFrame(differences).to_string(index=False,float_format=lambda x:f'{x:.4g}')]
    (OUTPUT_DIR/'report.txt').write_text('\n'.join(report)+'\n')
    config={'fit_csv':str(FIT_CSV),'manifest':str(MANIFEST),'primary_filter':PRIMARY_FILTER,
            'stability_tolerance':STABILITY_TOL,'bootstrap_reps':BOOTSTRAP_REPS,
            'run_mass_sensitivity':RUN_MASS_SENSITIVITY,
            'counts':{'n_stars':len(stars),'stable':int(stars.stable_mid.sum()),
                      'unstable':int((~stars.stable_mid).sum()),
                      'low_resolved':int(stars.low_resolved_all.sum()),
                      'stable_and_low_resolved':int((stars.stable_mid&stars.low_resolved_all).sum())}}
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2))
    print('\n'.join(report))
    print(f'\nSaved to {OUTPUT_DIR}')

if __name__=='__main__':main()
