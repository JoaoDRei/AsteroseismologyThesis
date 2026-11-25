#!/usr/bin/env python3
"""Matched-sample physical scaling regressions, one KIC per filter.
Run from project root: python Baseline/Granulation/analyze_granulation_physical_scaling_v4.py
Requires numpy, pandas, scipy, statsmodels, matplotlib.
All logarithms are base 10; exponential metallicity coefficient = beta*ln(10).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'Baseline/manifest.csv'
FIT_CSV = ROOT / 'data/fitted_parameters_v4_three_fixed_all/fit_results.csv'
GRAVITY_SOURCE = 'seis'  # 'seis' or 'spec'
GRAVITY_COLUMNS = {
    'seis': 'logg_seis',
    'spec': 'logg_spec',
}
OUT = ROOT / f'results/granulation_physical_scaling_v4_{GRAVITY_SOURCE}'
COLS = {
    'mass': 'mass',
    'radius': 'radius',
    'logg': GRAVITY_COLUMNS[GRAVITY_SOURCE],
}
FILTERS = ('55d','20d','80d')
PRIMARY = '55d'
MIN_N = 50
SEED = 42
N_FOLDS = 5
# Phase-adjusted fits are primary; pooled fits facilitate literature comparison.
PHASE_OPTIONS = (True, False)
SAMPLES = ('all_valid','stable_mid','stable_plausible','stable_and_low_resolved')
SOLAR = {'numax':3090., 'teff':5772., 'logg':4.438}

def positive(s):
    s = pd.to_numeric(s, errors='coerce')
    return s.where(np.isfinite(s) & (s>0))

def flag(s):
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes','1.0'])

def metadata():
    raw = pd.read_csv(MANIFEST, low_memory=False)
    source_cols = {c: COLS.get(c, c) for c in (
        'kic', 'nu_max', 'teff', 'metallicity',
        'evolstate', 'mass', 'radius', 'logg',
    )}
    gravity_source_cols = {}
    for source, column in GRAVITY_COLUMNS.items():
        actual_column = column
        if source == 'seis' and actual_column not in raw and 'logg' in raw:
            actual_column = 'logg'
        if actual_column not in raw:
            raise ValueError(
                f'Missing manifest gravity column {column!r}'
                + (" (legacy alias 'logg' is also absent)." if source == 'seis' else '.')
            )
        gravity_source_cols[source] = actual_column
    source_cols['logg'] = gravity_source_cols[GRAVITY_SOURCE]
    missing = set(source_cols.values()) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing manifest columns: {sorted(missing)}')
    d = pd.DataFrame({c: raw[source] for c, source in source_cols.items()})
    d['kic'] = pd.to_numeric(d.kic, errors='raise').astype('int64')
    for c in ['nu_max', 'teff', 'mass', 'radius']:
        d[c] = positive(d[c])
    for c in ['metallicity', 'logg']:
        d[c] = pd.to_numeric(d[c], errors='coerce').where(
            lambda x: np.isfinite(x) & (x > -999)
        )
    d['logg'] = d.logg.where(d.logg.between(0, 6))
    for source, column in gravity_source_cols.items():
        values = pd.to_numeric(raw[column], errors='coerce')
        d[f'logg_{source}_original'] = values.where(
            np.isfinite(values) & values.between(0, 6)
        )
    d['evolstate']=d.evolstate.astype('string').str.strip().str.upper()
    d['evolstate']=d.evolstate.where(d.evolstate.isin(['RGB','RC']))
    conflicts=d.groupby('kic').nunique(dropna=True).gt(1).sum()
    print('Manifest KICs with conflicting metadata:',conflicts.to_dict())
    if conflicts['evolstate']: raise ValueError('Conflicting RGB/RC labels; resolve before fitting')
    m=d.groupby('kic').median(numeric_only=True).join(
        d.groupby('kic').evolstate.first()
    ).reset_index()
    return m, conflicts.to_dict()

def load():
    d=pd.read_csv(FIT_CSV,low_memory=False)
    d=d[(d.model=='three_fixed') & flag(d.success)].copy()
    d['kic']=pd.to_numeric(d.kic,errors='raise').astype('int64')
    d['duration']=d.duration.astype(str).str.strip().str.lower()
    d=d[d.duration.isin(FILTERS)]
    if d.duplicated(['kic','duration']).any(): raise ValueError('Duplicate KIC/filter fits')
    for label in ['mid','high']:
        for short,suffix in [('b','component_nu_char_uHz'),('a','integrated_rms_native_sqrt_uHz')]:
            d[f'{short}_{label}']=positive(d[f'{label}_{suffix}'])
    nu=positive(d.nu_max_catalogue)
    d['plausible']= (d.b_mid/nu).between(.15,.60)&(d.b_high/nu).between(.50,1.80)&(d.b_mid/d.b_high).between(.20,.50)
    d['low_ok']=~(flag(d.low_char_below_fit_band)|flag(d.low_char_below_filter_timescale))
    g=d.groupby('kic'); q=g.duration.nunique().eq(3).to_frame('complete')
    for c in ['a_mid','b_mid']:
        q[c+'_spread']=(g[c].max()/g[c].min()-1).where(g[c].count().eq(3))
    q['stable_mid']=q.complete & q.a_mid_spread.le(.10)&q.b_mid_spread.le(.10)
    q['stable_plausible']=q.stable_mid & g.plausible.all()
    q['stable_and_low_resolved']=q.stable_mid & g.low_ok.all()
    m,conflicts=metadata(); d=d.merge(m,on='kic',how='left',validate='many_to_one').join(q,on='kic')
    d['delta_logg'] = d['logg_spec_original'] - d['logg_seis_original']
    d['all_valid']=True
    d['N']=np.log10(d.nu_max/SOLAR['numax']); d['T']=np.log10(d.teff/SOLAR['teff'])
    d['M']=np.log10(d.mass); d['R']=np.log10(d.radius); d['G']=d.logg-SOLAR['logg']; d['F']=d.metallicity
    d['RC']=d.evolstate.map({'RGB':0.,'RC':1.}).astype(float)
    d['gravity_consistency'] = d['G'] - (d['M'] - 2 * d['R'])
    d['seismic_gravity_consistency'] = (
        d['logg_seis_original'] - SOLAR['logg'] - (d['N'] + .5 * d['T'])
    )
    return d,q,conflicts

def design(d, predictors):
    return sm.add_constant(d[list(predictors)].astype(float),has_constant='add')

def fit(d,y,predictors,folds):
    X=design(d,predictors); y=np.asarray(y,float)
    if np.linalg.matrix_rank(X)<X.shape[1]: raise ValueError('Rank deficient design')
    r=sm.OLS(y,X).fit(cov_type='HC3')
    pred=np.empty(len(d))
    for k in range(N_FOLDS):
        test=folds==k; train=~test
        pred[test]=X.to_numpy()[test]@np.linalg.lstsq(X.to_numpy()[train],y[train],rcond=None)[0]
    z=X.iloc[:,1:].to_numpy(); z=(z-z.mean(0))/z.std(0)
    return r, float(np.sqrt(np.mean((y-pred)**2))),float(np.linalg.cond(z))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d,q,conflicts=load(); q.to_csv(OUT/'quality_flags.csv')
    d.to_csv(OUT/'merged_fits_metadata.csv',index=False)
    coverage=[]; results=[]; coefs=[]; checks=[]; nested=[]
    for duration in FILTERS:
      for sample in SAMPLES:
        raw=d[(d.duration==duration)&d[sample]].copy()
        required=[
            'N','T','M','R','G','F','RC','a_mid','a_high','b_mid','b_high',
            'logg_seis_original','logg_spec_original',
        ]
        common=raw.dropna(subset=required).copy().sort_values('kic')
        common=common[np.isfinite(common[required]).all(axis=1)]
        coverage.append(dict(duration=duration,sample=sample,n_raw=len(raw),n_common=len(common),**{f'missing_{c}':int(raw[c].isna().sum()) for c in required}))
        if len(common)<MIN_N: continue
        folds=np.random.default_rng(SEED).permutation(len(common))%N_FOLDS
        for phase in PHASE_OPTIONS:
          p=['RC'] if phase else []
          for component in ['mid','high']:
            for kind in ['amplitude','frequency']:
              base=dict(duration=duration,sample=sample,phase_adjusted=phase,component=component,kind=kind,n=len(common))
              y=np.log10(common[('a_' if kind=='amplitude' else 'b_')+component])
              amp=kind=='amplitude'
              specs={'A_numax':(['N'],0),'B_corsaro':(['N','M','F'],0),'C_extended':(['N','M','T','F'],0),'legacy_no_mass':(['N','T','F'],0),'D_mallick':(['G','T','F'],1 if amp else 0),'E_mallick_free_mass':(['G','T','F','M'],1 if amp else 0),'radius_only_change':(['N','T','F'],1 if amp else 0)}
              if not amp: specs.pop('radius_only_change')
              fitted={}
              for name,(xs,correct) in specs.items():
                target=y+correct*common.R
                try: r,cv,cond=fit(common,target,xs+p,folds)
                except (ValueError,np.linalg.LinAlgError) as e:
                    results.append(dict(**base,model=name,error=str(e))); continue
                fitted[name]=r
                # For radius-corrected models subtract known R to compare observed-amplitude predictions.
                obs_pred=r.fittedvalues-correct*common.R
                obs_rmse=float(np.sqrt(np.mean((y-obs_pred)**2)))
                obs_r2=1-np.sum((y-obs_pred)**2)/np.sum((y-y.mean())**2)
                results.append(dict(**base,model=name,radius_corrected=bool(correct),observed_rmse=obs_rmse,observed_r2=obs_r2,cv_observed_rmse=cv,aic=r.aic,bic=r.bic,condition_number=cond,error=''))
                for term in r.params.index:
                    lo,hi=r.conf_int().loc[term]
                    coefs.append(dict(**base,model=name,term=term,beta=r.params[term],se=r.bse[term],ci_low=lo,ci_high=hi,p=r.pvalues[term],exp_metallicity_coefficient=r.params[term]*np.log(10) if term=='F' else np.nan))
              for name,term,null in [('C_extended','M',-.5 if amp else 0),('C_extended','T',0),('E_mallick_free_mass','M',0),('D_mallick','T',0)]:
                if name not in fitted: continue
                r=fitted[name]; z=(r.params[term]-null)/r.bse[term]
                nested.append(dict(**base,model=name,term=term,null=null,beta=r.params[term],z=z,p=2*norm.sf(abs(z))))
              if amp and all(k in fitted for k in ['legacy_no_mass','radius_only_change']):
                rr,_,_=fit(common,common.R,['N','T','F']+p,folds)
                a=fitted['legacy_no_mass'].params.F; b=fitted['radius_only_change'].params.F
                checks.append(dict(**base,observed_beta=a,radius_beta=rr.params.F,corrected_beta=b,identity_error=b-a-rr.params.F))
    tables={'coverage':coverage,'model_comparison':results,'coefficients':coefs,'hypothesis_tests':nested,'radius_metallicity_decomposition':checks}
    for name,rows in tables.items(): pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    if not coefs: raise ValueError('No usable common samples; inspect coverage.csv and manifest units')
    c=pd.DataFrame(coefs); r=pd.DataFrame(results); h=pd.DataFrame(nested); ch=pd.DataFrame(checks)
    sel=lambda x:x[(x.duration==PRIMARY)&x.phase_adjusted] if len(x) else x
    mainc=sel(c); mainr=sel(r)
    report=['V4 EXTENDED PHYSICAL SCALING',f'Gravity source: {GRAVITY_SOURCE}','55d primary; other filters are correlated sensitivity checks.','Common complete-case sample per filter and quality cut; HC3 OLS; unadjusted exploratory tests.','Native amplitudes are not assumed calibrated ppm. Radius correction is a geometric proxy.','Mass/radius/gravity may share seismic inputs; tests are conditional associations, not causal.','AIC/BIC and CV errors use equivalent observed-log targets (radius is a known offset).','Metadata conflicts (median used for numeric fields): '+str(conflicts),'Gravity consistency residual quantiles: '+str(d[['gravity_consistency','seismic_gravity_consistency']].quantile([.16,.5,.84]).to_dict()),'COVERAGE',pd.DataFrame(coverage).to_string(index=False),'PRIMARY METALLICITY COEFFICIENTS',mainc[mainc.term=='F'].to_string(index=False),'PRIMARY MODEL COMPARISON',mainr.to_string(index=False),'PRIMARY HYPOTHESIS TESTS',sel(h).to_string(index=False),'RADIUS DECOMPOSITION',sel(ch).to_string(index=False)]
    (OUT/'report.txt').write_text('\n\n'.join(report))
    for kind in ['amplitude','frequency']:
      fig,ax=plt.subplots(figsize=(10,5)); sub=mainc[(mainc['kind']==kind)&(mainc['sample']=='all_valid')&(mainc['term']=='F')]
      for i,comp in enumerate(['mid','high']):
        t=sub[sub.component==comp]; xx=np.arange(len(t))+(i-.5)*.18
        ax.errorbar(xx,t.beta,yerr=[t.beta-t.ci_low,t.ci_high-t.beta],fmt='o',label=comp)
      if len(t): ax.set_xticks(np.arange(len(t)),t.model,rotation=30,ha='right')
      ax.axhline(0,color='black',lw=1); ax.set_ylabel('Metallicity slope (log10 per dex)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/f'{kind}_metallicity.png',dpi=150); plt.close(fig)
    (OUT/'run_config.json').write_text(json.dumps(dict(manifest=str(MANIFEST),fit_csv=str(FIT_CSV),columns=COLS,gravity_source=GRAVITY_SOURCE,gravity_columns=GRAVITY_COLUMNS,solar=SOLAR,filters=FILTERS,primary=PRIMARY,folds=N_FOLDS,seed=SEED),indent=2))
    print('Saved:',OUT); print(mainc[(mainc['term']=='F')&(mainc['sample']=='all_valid')].to_string(index=False))

if __name__=='__main__': main()
