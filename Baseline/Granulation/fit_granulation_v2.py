#!/usr/bin/env python3
"""Catalogue-constrained two-component Harvey background fitter.

Run from any directory with defaults:
 python path/to/fit_granulation_v2.py

Uses physical PSD .npz files, NOT manifest psd_path .npy files.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "Baseline" / "manifest.csv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "psd_physical"
MAX_STARS = 0 #0 for full sample
FIT_ALPHA = True
SAMPLE_SEED = 42


def parse_filename(path):
    m = re.search(r'kic0*(\d+).*?(20|55|80)d', path.stem, flags=re.I)
    if not m:
        raise ValueError(f'Cannot identify KIC and duration from {path.name}')
    return int(m.group(1)), f'{m.group(2)}d'


def load_manifest(path, tolerance=0.01):
    df = pd.read_csv(path, usecols=['kic','duration','nu_max'])
    df['kic'] = pd.to_numeric(df['kic'], errors='raise').astype('int64')
    df['nu_max'] = pd.to_numeric(df['nu_max'], errors='coerce')
    df['duration'] = df['duration'].astype(str).str.strip().str.lower()
    df = df[np.isfinite(df.nu_max) & (df.nu_max > 0)].copy()
    if df.empty:
        raise ValueError('No valid positive nu_max values in manifest')
    # Verify repeated catalogue values for the same KIC; allow small rounding differences.
    grouped = df.groupby('kic')['nu_max']
    inconsistent = grouped.agg(lambda x: (x.max()-x.min())/x.median() > tolerance)
    if inconsistent.any():
        raise ValueError(f'Inconsistent nu_max for KICs: {inconsistent[inconsistent].index.tolist()[:15]}')
    return grouped.median().to_dict(), set(zip(df.kic, df.duration))


def log_bin(freq, power, bins=180, min_points=4):
    edges = np.geomspace(freq.min(), freq.max(), bins+1)
    ix = np.searchsorted(edges, freq, side='right')-1
    xs, ys, counts = [], [], []
    for j in range(bins):
        m = ix == j
        if m.sum() >= min_points:
            xs.append(np.exp(np.mean(np.log(freq[m]))))
            ys.append(np.median(power[m]))
            counts.append(int(m.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(counts)


def harvey(nu, height, char_freq, alpha):
    return height / (1 + (nu/char_freq)**alpha)


def unpack(theta, fit_alpha):
    if fit_alpha:
        h1, c1, a1, h2, c2, a2, w, g, lw = theta
    else:
        h1, c1, h2, c2, w, g, lw = theta
        a1 = a2 = 4.0
    return np.exp(h1), np.exp(c1), a1, np.exp(h2), np.exp(c2), a2, np.exp(w), np.exp(g), np.exp(lw)


def evaluate(theta, nu, nu_max, fit_alpha):
    H1, n1, a1, H2, n2, a2, W, G, width = unpack(theta, fit_alpha)
    c1 = harvey(nu, H1, n1, a1)
    c2 = harvey(nu, H2, n2, a2)
    envelope = G*np.exp(-0.5*(np.log(nu/nu_max)/width)**2)
    return c1+c2+W+envelope, (c1,c2,np.full_like(nu,W),envelope)


def amplitude_proxy(H, n, alpha):
    return float(np.sqrt(H*n*np.pi/(alpha*np.sin(np.pi/alpha))))


def fit_one(path, nu_max, plot_path, args):
    with np.load(path) as d:
        freq = np.asarray(d['frequency_uHz'],float)
        power = np.asarray(d['power'],float)
    good = np.isfinite(freq)&np.isfinite(power)&(freq>=args.min_frequency)&(freq<=args.max_frequency)&(power>0)
    freq,power=freq[good],power[good]
    if len(freq)<100:
        raise ValueError('Fewer than 100 valid PSD samples')
    order=np.argsort(freq)
    freq,power=freq[order],power[order]
    xb,yb,counts=log_bin(freq,power,args.bins,args.min_bin_points)
    if len(xb)<30:
        raise ValueError('Fewer than 30 populated logarithmic bins')
    fmin,fmax=float(xb.min()),float(xb.max())
    if not fmin<nu_max<fmax:
        raise ValueError(f'Catalogue nu_max={nu_max:.3f} outside binned PSD [{fmin:.3f},{fmax:.3f}]')
    ymax=float(np.percentile(yb,90))
    ymin=max(float(np.percentile(yb,10)*0.1),np.finfo(float).tiny)
    # Keep the two characteristic frequencies initially distinct, within the fitted band.
    n1=np.clip(nu_max/20,fmin*1.1,fmax/1.1)
    n2=np.clip(nu_max/3,fmin*1.1,fmax/1.1)
    # Fit the oscillation amplitude, but fix its centre at catalogue nu_max.
    base=[np.log(max(ymax*0.7,ymin)),np.log(n1),
          np.log(max(ymax*0.3,ymin)),np.log(n2),
          np.log(ymin),np.log(max(ymax*0.3,ymin)),np.log(0.35)]
    lo=[np.log(ymin*1e-3),np.log(fmin/10),
        np.log(ymin*1e-3),np.log(fmin/10),
        np.log(ymin*1e-4),np.log(ymin*1e-3),np.log(0.08)]
    hi=[np.log(ymax*1e3),np.log(fmax*10),
        np.log(ymax*1e3),np.log(fmax*10),
        np.log(ymax*10),np.log(ymax*1e3),np.log(0.8)]
    if args.fit_alpha:
        p0=np.array([base[0],base[1],4.,base[2],base[3],4.,*base[4:]])
        lower=np.array([lo[0],lo[1],2.,lo[2],lo[3],2.,*lo[4:]])
        upper=np.array([hi[0],hi[1],6.,hi[2],hi[3],6.,*hi[4:]])
    else:
        p0=np.array(base)
        lower=np.array(lo)
        upper=np.array(hi)
    p0=np.clip(p0,lower+1e-8,upper-1e-8)
    def residual(theta):
        pred,_=evaluate(theta,xb,nu_max,args.fit_alpha)
        return np.log10(np.maximum(pred,np.finfo(float).tiny))-np.log10(yb)
    result=least_squares(residual,p0,bounds=(lower,upper),loss='soft_l1',f_scale=0.15,max_nfev=args.max_nfev)
    if not result.success:
        raise RuntimeError(result.message)
    H1,n1,a1,H2,n2,a2,W,G,width=unpack(result.x,args.fit_alpha)
    (Hl,nl,al),(Hs,ns,ass)=sorted([(H1,n1,a1),(H2,n2,a2)],key=lambda t:t[1])
    pred,_=evaluate(result.x,xb,nu_max,args.fit_alpha)
    rms=float(np.sqrt(np.mean((np.log10(pred)-np.log10(yb))**2)))
    kic,duration=parse_filename(path)
    if args.plot:
        plot_path.parent.mkdir(parents=True,exist_ok=True)
        dense=np.geomspace(freq.min(),freq.max(),1000)
        total,components=evaluate(result.x,dense,nu_max,args.fit_alpha)
        lower=max(float(min(np.min(yb),np.min(total))*0.5),np.finfo(float).tiny)
        upper=float(max(np.max(yb),np.max(total))*2.0)
        fig,(ax,residual_ax)=plt.subplots(
            2,1,figsize=(8,7),sharex=True,
            gridspec_kw={"height_ratios":[3,1]},
        )
        ax.loglog(freq,power,'.',alpha=.12,ms=1,label='PSD')
        ax.loglog(xb,yb,'.',ms=4,label='Binned median')
        ax.loglog(dense,total,label='Fit',lw=2)
        for comp,label in zip(components,['Harvey 1','Harvey 2','White','Envelope']):
            visible=np.where(
                np.isfinite(comp)&(comp>=lower)&(comp<=upper),
                comp,
                np.nan,
            )
            if np.isfinite(visible).any():
                ax.loglog(dense,visible,'--',label=label,alpha=.75)
        ax.axvline(nu_max,color='k',ls=':',label=r'Catalogue $\nu_{\max}$')
        ax.set_ylim(lower,upper)
        ax.set_ylabel('PSD (native units)')
        ax.set_title(f'KIC {kic} ({duration}) | log-RMS={rms:.3f} dex')
        ax.legend(fontsize=8)
        ax.grid(True,which='both',alpha=0.2)
        residual_ax.semilogx(xb,np.log10(yb)-np.log10(pred),'o',ms=3)
        residual_ax.axhline(0,color='k',lw=1)
        residual_ax.set(xlabel='Frequency (microHz)',ylabel='log10(data / fit)')
        residual_ax.grid(True,which='both',alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_path,dpi=130)
        plt.close(fig)
    return dict(file=path.name,kic=kic,duration=duration,nu_max_catalogue=nu_max,
                success=True,cost=float(result.cost),rms_log10_residual=rms,
                long_component_height=Hl,long_component_nu_char_uHz=nl,
                long_component_tau_days=1/(2*np.pi*nl*1e-6*86400),long_component_alpha=al,
                short_component_height=Hs,short_component_nu_char_uHz=ns,
                short_component_tau_days=1/(2*np.pi*ns*1e-6*86400),short_component_alpha=ass,
                long_integrated_rms_native_sqrt_uHz=amplitude_proxy(Hl,nl,al),
                short_integrated_rms_native_sqrt_uHz=amplitude_proxy(Hs,ns,ass),
                white_noise=W,oscillation_envelope_amplitude=G,
                oscillation_envelope_centre_uHz=nu_max,oscillation_envelope_log_width=width,
                n_raw_points=len(freq),n_fit_bins=len(xb),message=str(result.message))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--plot-dir',type=Path,default=None)
    parser.add_argument(
        '--max-stars',
        type=int,
        default=MAX_STARS,
        help=f'Unique KICs; 0 means all (default: MAX_STARS={MAX_STARS})',
    )
    alpha_options=parser.add_mutually_exclusive_group()
    alpha_options.add_argument(
        '--fit-alpha',dest='fit_alpha',action='store_true',
        help='Fit alpha in [2,6]',
    )
    alpha_options.add_argument(
        '--fixed-alpha',dest='fit_alpha',action='store_false',
        help='Fix both Harvey slopes at 4',
    )
    parser.set_defaults(fit_alpha=FIT_ALPHA)
    plot_options=parser.add_mutually_exclusive_group()
    plot_options.add_argument('--plot',dest='plot',action='store_true',help='Save individual diagnostic plots (default)')
    plot_options.add_argument('--no-plot',dest='plot',action='store_false',help='Do not save individual diagnostic plots')
    parser.set_defaults(plot=True)
    parser.add_argument('--min-frequency',type=float,default=0.1)
    parser.add_argument('--max-frequency',type=float,default=283.0)
    parser.add_argument('--bins',type=int,default=180)
    parser.add_argument('--min-bin-points',type=int,default=4)
    parser.add_argument('--max-nfev',type=int,default=12000)
    args=parser.parse_args()
    if args.max_stars<0: parser.error('--max-stars must be >= 0')
    sample_label=str(args.max_stars) if args.max_stars else "all"
    alpha_label="free_alpha" if args.fit_alpha else "fixed_alpha"
    suffix=f"_{sample_label}_{alpha_label}"
    args.output_dir=args.output_dir or PROJECT_ROOT/"data"/f"fitted_parameters_v2{suffix}"
    args.plot_dir=args.plot_dir or PROJECT_ROOT/"results"/f"granulation_v2_plots{suffix}"
    lookup,manifest_pairs=load_manifest(args.manifest)
    files=[]
    for path in sorted(args.input_dir.glob('*.npz')):
        try: kic,duration=parse_filename(path)
        except ValueError: continue
        if kic in lookup and (kic,duration) in manifest_pairs:
            files.append((path,kic,duration))
    if not files: raise ValueError('No physical PSD filenames match manifest KIC/duration rows')
    if args.max_stars:
        available_kics=np.asarray(sorted({kic for _,kic,_ in files}),dtype=np.int64)
        sample_size=min(args.max_stars,len(available_kics))
        rng=np.random.default_rng(SAMPLE_SEED)
        selected=set(rng.choice(available_kics,size=sample_size,replace=False).tolist())
        files=[item for item in files if item[1] in selected]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results,failures=[],[]
    for i,(path,kic,duration) in enumerate(files,1):
        try:
            row=fit_one(path,lookup[kic],args.plot_dir/f'{path.stem}.png',args)
            results.append(row)
        except Exception as exc:
            failures.append(dict(file=path.name,kic=kic,duration=duration,error=repr(exc)))
        if i%100==0 or i==len(files):
            print(f'{i}/{len(files)} spectra | success={len(results)} failure={len(failures)}',flush=True)
            pd.DataFrame(results).to_csv(args.output_dir/'fit_results.csv',index=False)
            pd.DataFrame(failures,columns=['file','kic','duration','error']).to_csv(args.output_dir/'fit_failures.csv',index=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['sample_seed']=SAMPLE_SEED
    config['alpha_bounds']=[2.0,6.0] if args.fit_alpha else None
    config['envelope_width_bounds']=[0.08,0.8]
    with open(args.output_dir/'run_config.json','w') as f:
        json.dump(config,f,indent=2)
    print(f'Finished. Output: {args.output_dir.resolve()}')

if __name__=='__main__':
    main()
