import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def plot_comprehensive_diagnostic(kic_id, base_dir="./kepseismic_data_APOKASC3", proc_dir="./processed_ml_data_APOKASC3", catalog_path="./APOKASC-3_catalog.txt", filter_type="80d"):
    kic_padded = str(int(kic_id)).zfill(9)

    # --- 1. Load Official Catalog Values using Fixed-Width Specs ---
    v_max_lit, d_nu_lit, feh_lit, mass_lit = None, None, None, None
    try:
        # Using the (DocStart - 1, DocEnd) conversion rule
        specs = [
            (0, 8),     # KIC (1-8)
            (49, 59),   # Numax (50-59)
            (71, 81),   # DNu (72-81)
            (126, 136), # Mass (127-136)
            (236, 246)  # [Fe/H] (237-246)
        ]
        names = ['KIC', 'Numax', 'DNu', 'Mass', 'FeH']
        
        # Read full catalog
        df_cat = pd.read_fwf(catalog_path, colspecs=specs, names=names, skiprows=111)
        
        # Find our specific star
        star_data = df_cat[df_cat['KIC'] == int(kic_id)]
        
        if not star_data.empty:
            v_max_lit = star_data.iloc[0]['Numax']
            d_nu_lit = star_data.iloc[0]['DNu']
            feh_lit = star_data.iloc[0]['FeH']
            mass_lit = star_data.iloc[0]['Mass']
            
            # Handle the catalog's null value (-9999.0)
            if v_max_lit == -9999.0: v_max_lit = None
            if d_nu_lit == -9999.0: d_nu_lit = None
    except Exception as e:
        print(f"Catalog Load Error: {e}")

    # --- 2. Data Retrieval (FITS and NPY) ---
    fits_pattern = os.path.join(base_dir, "**", f"*{kic_padded}*{filter_type}*filt-inp.fits")
    fits_files = glob.glob(fits_pattern, recursive=True)
    lc_npy_path = os.path.join(proc_dir, f"kplr{kic_padded}_{filter_type}_clean.npy")
    psd_npy_path = os.path.join(proc_dir, f"kplr{kic_padded}_{filter_type}_psd.npy")

    # --- 3. Plotting Setup ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

    # PANEL 1: Raw KEPSEISMIC FITS
    if fits_files:
        lc = lk.read(fits_files[0])
        lc.plot(ax=ax1, color='gray', lw=0.5, label='Raw KEPSEISMIC (PPM)')
        ax1.set_title(f"KIC {kic_id} Diagnostic - {filter_type} Data")
    else:
        ax1.text(0.5, 0.5, f"FITS for {kic_id} Not Found", ha='center')

    # PANEL 2: Standardized ML Input
    if os.path.exists(lc_npy_path):
        data_lc = np.load(lc_npy_path)
        ax2.plot(data_lc[0], data_lc[1], color='black', lw=0.5)
        ax2.set_title("Astroconformer Input (Normalized Time-Series)")
        ax2.set_ylabel("Std Flux")
    else:
        ax2.text(0.5, 0.5, "Time-domain NPY Not Found", ha='center')

    # PANEL 3: Raw Power Spectrum
    if fits_files:
        pg = lc.to_periodogram(method='lombscargle', normalization='psd')
        pg.plot(ax=ax3, color='gray', lw=0.8, alpha=0.5)
        ax3.set_xscale('log'); ax3.set_yscale('log')
        ax3.set_xlim(0.01, 5000)
        ax3.set_title("Power Density Spectrum (Log-Log)")

    # PANEL 4: Processed PSD + Seismology Check
    if os.path.exists(psd_npy_path):
        p_freq, p_power = np.load(psd_npy_path)
        
        # --- CALCULATION LOGIC ---
        # Weighting power by freq^2 to flatten granulation and find the oscillation hump
        search_mask = (p_freq > 5.0) & (p_freq < 1000.0)
        f_search = p_freq[search_mask]
        weighted_p = p_power[search_mask] * (f_search**2)
        smoothed_p = gaussian_filter1d(weighted_p, sigma=20) 
        our_v_max = f_search[np.argmax(smoothed_p)]
        our_d_nu = 0.22 * (our_v_max**0.77) # Standard Scaling Relation

        # Terminal Printout for Thesis Validation
        print(f"\n" + "="*40)
        print(f"VALIDATION FOR KIC {kic_id}")
        print(f"="*40)
        print(f"SEISMIC:")
        print(f"  v_max: Catalog = {v_max_lit if v_max_lit else 'N/A':>7.2f} | Calc = {our_v_max:>7.2f}")
        print(f"  D_nu:  Catalog = {d_nu_lit if d_nu_lit else 'N/A':>7.2f} | Calc = {our_d_nu:>7.2f}")
        print(f"PHYSICAL/CHEMICAL:")
        print(f"  Mass:  {mass_lit if mass_lit else 'N/A':>7.2f} M_sun")
        print(f"  [Fe/H]: {feh_lit if feh_lit else 'N/A':>7.2f} (Metallicity)")
        print(f"="*40 + "\n")

        ax4.loglog(p_freq, p_power, color='royalblue', lw=0.8)
        ax4.axvline(our_v_max, color='green', lw=2, label=f'Calc v_max: {our_v_max:.1f}')
        if v_max_lit:
            ax4.axvline(v_max_lit, color='red', linestyle='--', alpha=0.6, label=f'Lit v_max: {v_max_lit:.1f}')
        ax4.set_xlim(0.01, 5000); ax4.set_xlabel("Frequency (µHz)")
        ax4.legend()
        
        # Secondary Figure: Linear Zoom of Oscillation Hump
        plt.figure(figsize=(10, 4))
        zoom_mask = (p_freq > our_v_max - (our_d_nu*5)) & (p_freq < our_v_max + (our_d_nu*5))
        plt.plot(p_freq[zoom_mask], p_power[zoom_mask], color='royalblue')
        # Draw Delta-Nu Spacing Grid
        for i in range(-3, 4):
            plt.axvline(our_v_max + (i*our_d_nu), color='black', alpha=0.2, ls=':')
        plt.title(f"Oscillation Peak Spacing Check (D_nu ≈ {our_d_nu:.2f})")
        plt.xlabel("Frequency (µHz)")
        
    plt.tight_layout()
    plt.show()

# Run for a Red Giant from APOKASC-3
plot_comprehensive_diagnostic(12554802)