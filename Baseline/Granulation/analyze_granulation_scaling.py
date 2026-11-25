#!/usr/bin/env python3
"""Explore APOKASC-3 [Fe/H] associations with V2 Harvey background fits.

Run from thesis root:
 python analyze_granulation_scaling.py
 python analyze_granulation_scaling.py --manifest Baseline/manifest.csv \
   --fits data/fitted_parameters_v2_all_free_alpha/fit_results.csv

Requires: numpy, pandas, scipy, matplotlib. Outputs are descriptive, not causal.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm

DURATIONS = ('20d', '55d', '80d')
TARGETS = {
    'short_component_nu_char_uHz': 'Short characteristic frequency (µHz)',
    'short_integrated_rms_native_sqrt_uHz': 'Short integrated RMS (native units)',
    'short_component_alpha': 'Short Harvey exponent',
    'long_component_nu_char_uHz': 'Long characteristic frequency (µHz)',
    'long_integrated_rms_native_sqrt_uHz': 'Long integrated RMS (native units)',
}


def clean_numeric(s):
    x = pd.to_numeric(s, errors='coerce')
    return x.mask((x <= -999) | ~np.isfinite(x))


def summarize_unique_manifest(df):
    needed = ['kic', 'nu_max', 'teff', 'metallicity', 'evolstate']
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f'Manifest missing columns: {sorted(missing)}')
    df = df[needed].copy()
    df['kic'] = pd.to_numeric(df.kic, errors='coerce')
    df = df.dropna(subset=['kic'])
    df['kic'] = df.kic.astype('int64')
    for col in ['nu_max', 'teff', 'metallicity']:
        df[col] = clean_numeric(df[col])
    df.loc[df.nu_max <= 0, 'nu_max'] = np.nan
    df.loc[df.teff <= 0, 'teff'] = np.nan
    df['evolstate'] = df.evolstate.astype('string').str.strip()
    # Identify discordant metadata instead of silently taking an arbitrary row.
    discordant = []
    for col in ['nu_max', 'teff', 'metallicity', 'evolstate']:
        counts = df.groupby('kic')[col].nunique(dropna=True)
        bad = counts[counts > 1]
        if len(bad):
            discordant.append(f'{col}: {len(bad)} KICs')
    if discordant:
        warnings.warn('Conflicting manifest values across durations: ' + ', '.join(discordant))
    # For conflicts, median numerical values and mode of state; report warning above.
    numeric = df.groupby('kic', as_index=False)[['nu_max','teff','metallicity']].median()
    states = (df.groupby('kic').evolstate.agg(lambda s: s.dropna().mode().iloc[0]
              if not s.dropna().mode().empty else pd.NA).reset_index())
    return numeric.merge(states, on='kic', validate='one_to_one')


def make_fit_table(fits):
    required = ['kic','duration'] + list(TARGETS)
    missing = set(required) - set(fits.columns)
    if missing:
        raise ValueError(f'Fits missing columns: {sorted(missing)}')
    fits = fits.copy()
    fits['kic'] = pd.to_numeric(fits.kic, errors='coerce')
    fits = fits.dropna(subset=['kic'])
    fits['kic'] = fits.kic.astype('int64')
    fits['duration'] = fits.duration.astype(str).str.strip().str.lower().str.replace('days','d',regex=False)
    fits['duration'] = fits.duration.str.replace(r'^(20|55|80)$',r'\1d',regex=True)
    fits = fits[fits.duration.isin(DURATIONS)].copy()
    if fits.duplicated(['kic','duration']).any():
        raise ValueError('Duplicate KIC-duration rows in fit_results.csv; resolve before proceeding.')
    for col in TARGETS:
        fits[col] = clean_numeric(fits[col])
    for col in ['long_component_alpha','short_component_alpha','oscillation_envelope_log_width',
                'rms_log10_residual','short_component_height','long_component_height']:
        if col in fits:
            fits[col] = clean_numeric(fits[col])
    if 'success' in fits:
        fits = fits[fits.success.astype(str).str.lower().isin(['true','1'])].copy()
    fits['alpha_bound'] = False
    for col in ['long_component_alpha','short_component_alpha']:
        if col in fits:
            fits['alpha_bound'] |= (np.isclose(fits[col],2,atol=.01,rtol=0) |
                                    np.isclose(fits[col],6,atol=.01,rtol=0))
    fits['width_bound'] = False
    if 'oscillation_envelope_log_width' in fits:
        w = fits.oscillation_envelope_log_width
        fits['width_bound'] = np.isclose(w,.08,atol=.01,rtol=0) | np.isclose(w,.8,atol=.01,rtol=0)
    fits['any_bound'] = fits.alpha_bound | fits.width_bound
    return fits


def safe_spearman(x,y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return (np.nan,np.nan)
    result = spearmanr(x[valid],y[valid])
    return float(result.statistic),float(result.pvalue)


def robust_ols(y, X, names):
    """OLS with heteroskedasticity-consistent HC3 standard errors."""
    y = np.asarray(y,dtype=float)
    X = np.asarray(X,dtype=float)
    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    resid = y-X@beta
    n,p = X.shape
    rank = np.linalg.matrix_rank(X)
    if rank < p or n <= p+2:
        raise ValueError('Regression design rank deficient or insufficient observations')
    inv = np.linalg.inv(X.T@X)
    leverage = np.einsum('ij,jk,ik->i',X,inv,X)
    weights = (resid / np.maximum(1-leverage,1e-8))**2
    cov = inv @ (X.T @ (X * weights[:,None])) @ inv
    se = np.sqrt(np.maximum(np.diag(cov),0))
    z = np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    pvals = 2*norm.sf(np.abs(z))
    ss_total = np.sum((y-y.mean())**2)
    r2 = 1-np.sum(resid**2)/ss_total if ss_total>0 else np.nan
    return {'n':int(n),'r_squared':float(r2),
            'coefficients':{name:{'estimate':float(b),'hc3_se':float(s),
            'ci95_lower':float(b-1.96*s),'ci95_upper':float(b+1.96*s),
            'p_value_normal_approx':float(pv)}
            for name,b,s,pv in zip(names,beta,se,pvals)}},resid


def prepare_regression(df, target):
    cols = ['nu_max','teff','metallicity','evolstate',target]
    t = df[cols].copy().dropna()
    t = t[(t.nu_max>0)&(t.teff>0)&(t[target]>0)&t.evolstate.isin(['RGB','RC'])]
    t['log_nu'] = np.log10(t.nu_max / 40.)
    t['log_teff'] = np.log10(t.teff / 4800.)
    t['is_rc'] = (t.evolstate=='RC').astype(float)
    t['log_y'] = np.log10(t[target])
    return t


def plot_scaling(df, target, out):
    t = df[['nu_max','evolstate',target]].dropna()
    t = t[(t.nu_max>0)&(t[target]>0)&t.evolstate.isin(['RGB','RC'])]
    if t.empty:
        return
    fig,ax = plt.subplots(figsize=(7.2,5.4))
    for state,color in [('RGB','tab:blue'),('RC','tab:orange')]:
        s=t[t.evolstate==state]
        ax.scatter(s.nu_max,s[target],s=5,alpha=.22,color=color,label=f'{state} (n={len(s)})',rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Catalogue $\nu_{\max}$ (µHz)'); ax.set_ylabel(TARGETS[target]); ax.legend(markerscale=2)
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out,dpi=170); plt.close(fig)


def plot_metallicity(df, target, out):
    t=df[['metallicity','evolstate',target]].dropna()
    t=t[(t[target]>0)&t.evolstate.isin(['RGB','RC'])]
    if t.empty:return
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for state,color in [('RGB','tab:blue'),('RC','tab:orange')]:
        s=t[t.evolstate==state]
        ax.scatter(s.metallicity,np.log10(s[target]),s=5,alpha=.22,color=color,label=state,rasterized=True)
    ax.set_xlabel('[Fe/H] (dex)');ax.set_ylabel('log10 '+TARGETS[target]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def regress(df, target, outdir, tag):
    t=prepare_regression(df,target)
    if len(t)<30:
        return {'n':int(len(t)),'error':'Too few complete cases'}
    baseline_names=['intercept','log10(nu_max/40uHz)','log10(teff/4800K)','RC_indicator']
    X=np.column_stack([np.ones(len(t)),t.log_nu,t.log_teff,t.is_rc])
    try:
        baseline,resid=robust_ols(t.log_y,X,baseline_names)
        extended,ext_resid=robust_ols(t.log_y,np.column_stack([X,t.metallicity]),baseline_names+['[Fe/H]'])
    except ValueError as e:
        return {'n':int(len(t)),'error':str(e)}
    rho,p=safe_spearman(t.metallicity.to_numpy(),resid)
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for state,color in [('RGB','tab:blue'),('RC','tab:orange')]:
        mask=(t.evolstate==state).to_numpy()
        ax.scatter(t.metallicity.to_numpy()[mask],resid[mask],s=6,alpha=.25,color=color,label=state,rasterized=True)
    ax.axhline(0,color='black',lw=1,alpha=.5)
    ax.set_xlabel('[Fe/H] (dex)');ax.set_ylabel('Baseline log10 amplitude residual (dex)')
    ax.set_title(f'{tag}: baseline residual vs [Fe/H]');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(outdir/f'{tag}_metallicity_residual.png',dpi=170);plt.close(fig)
    t['baseline_residual_dex']=resid
    t['extended_residual_dex']=ext_resid
    t.to_csv(outdir/f'{tag}_regression_rows.csv',index=False)
    return {'n':len(t),'baseline':baseline,'with_metallicity':extended,
            'delta_r_squared':float(extended['r_squared']-baseline['r_squared']),
            'baseline_residual_spearman_with_fe_h':{'rho':rho,'p':p}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default='Baseline/manifest.csv')
    ap.add_argument('--fits',default='data/fitted_parameters_v2_all_free_alpha/fit_results.csv')
    ap.add_argument('--outdir',default='results/granulation_scaling_v2')
    ap.add_argument('--primary-duration',choices=DURATIONS,default='55d')
    args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    meta=summarize_unique_manifest(pd.read_csv(args.manifest,low_memory=False))
    fits=make_fit_table(pd.read_csv(args.fits,low_memory=False))
    merged=fits.merge(meta,on='kic',how='left',validate='many_to_one',indicator=True)
    merged['valid_stellar_metadata']=(merged.nu_max.gt(0)&merged.teff.gt(0)&
                                     merged.metallicity.notna()&merged.evolstate.isin(['RGB','RC']))
    merged.to_csv(out/'all_durations_merged.csv',index=False)
    report={'input':vars(args),'total_fit_rows':int(len(fits)),'unique_fit_kics':int(fits.kic.nunique()),
            'unmatched_fit_rows':int((merged._merge=='left_only').sum()),
            'metallicity_missing_rows':int(merged.metallicity.isna().sum()),
            'ambiguous_state_rows':int((~merged.evolstate.isin(['RGB','RC'])).sum()),
            'durations':{}}
    summary=[]
    for duration in DURATIONS:
        sub=merged[merged.duration==duration].copy()
        valid=sub[sub.valid_stellar_metadata].copy()
        report['durations'][duration]={'n_fits':int(len(sub)),'n_valid_stellar':int(len(valid)),
                                       'n_alpha_bound':int(valid.alpha_bound.sum()),
                                       'n_any_bound':int(valid.any_bound.sum()),'analyses':{}}
        valid.to_csv(out/f'stars_{duration}_all_valid.csv',index=False)
        filtered=valid[~valid.any_bound].copy()
        filtered.to_csv(out/f'stars_{duration}_no_bounds.csv',index=False)
        for target in TARGETS:
            plot_scaling(valid,target,out/f'{duration}_{target}_vs_nu_max.png')
            if target.startswith('short_'):
                plot_metallicity(valid,target,out/f'{duration}_{target}_vs_fe_h.png')
            for name,sample in [('all_valid',valid),('no_bounds',filtered)]:
                tag=f'{duration}_{name}_{target}'
                result=regress(sample,target,out,tag)
                report['durations'][duration]['analyses'][f'{name}_{target}']=result
                if 'with_metallicity' in result:
                    c=result['with_metallicity']['coefficients']['[Fe/H]']
                    summary.append({'duration':duration,'sample':name,'target':target,'n':result['n'],
                                    'feh_coefficient':c['estimate'],'feh_hc3_se':c['hc3_se'],
                                    'feh_ci95_lower':c['ci95_lower'],'feh_ci95_upper':c['ci95_upper'],
                                    'feh_p_value':c['p_value_normal_approx'],
                                    'baseline_r2':result['baseline']['r_squared'],
                                    'extended_r2':result['with_metallicity']['r_squared'],
                                    'delta_r2':result['delta_r_squared']})
    pd.DataFrame(summary).to_csv(out/'regression_summary.csv',index=False)
    (out/'analysis_summary.json').write_text(json.dumps(report,indent=2,allow_nan=True,default=str))
    lines=[f"V2 GRANULATION SCALING — primary duration {args.primary_duration}",
           f"Fit rows: {len(fits):,}; unique KICs: {fits.kic.nunique():,}",
           f"Unmatched fit rows: {report['unmatched_fit_rows']}",
           'Metallicity is APOKASC-3 [Fe/H]; -9999 treated as missing.',
           'OLS uncertainty: HC3 heteroskedasticity-robust SE, normal-approximation p-values.',
           'Exploratory associations; no measurement-error or selection-function model.',
           'Native integrated RMS is NOT verified ppm; integral extrapolates beyond observed band.',
           'No-bound selection may induce selection bias; compare both samples.', '']
    for duration in DURATIONS:
        info=report['durations'][duration]
        lines.append(f"{duration}: {info['n_fits']:,} fits; {info['n_valid_stellar']:,} valid stellar rows; {info['n_any_bound']:,} flagged bounds")
    lines.append('\nMETALLICITY COEFFICIENTS (log10 target per dex [Fe/H]):')
    if summary:
        tab=pd.DataFrame(summary)
        tab=tab[(tab.duration==args.primary_duration)&tab.target.isin(['short_integrated_rms_native_sqrt_uHz','short_component_nu_char_uHz'])]
        lines.append(tab.to_string(index=False,float_format=lambda x:f'{x:.4g}'))
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f'\nOutputs: {out.resolve()}')

if __name__=='__main__':
    main()
