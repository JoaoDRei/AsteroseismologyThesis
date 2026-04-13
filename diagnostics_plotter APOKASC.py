import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def plot_comprehensive_diagnostic(kic_id, base_dir="./kepseismic_data_APOKASC3", proc_dir="./processed_ml_data_APOKASC3", catalog_path="./table4APOKASC.txt", filter_type="80d"):
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

    # FIXING OVERLAP: Increase vertical spacing between subplots
    plt.subplots_adjust(hspace=0.4)

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
        ax2.set_title("Normalized Time-Series")
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
        weighted_p = p_power[search_mask] * (f_search**2) #flattens background to highlight the hump
        smoothed_p = gaussian_filter1d(weighted_p, sigma=20) #removes spikes to find the hump more clearly
        our_v_max = f_search[np.argmax(smoothed_p)]
        our_d_nu = 0.22 * (our_v_max**0.77) # Standard Scaling Relation

        # Terminal Printout for Validation
        print(f"\n" + "="*40)
        print(f"VALIDATION FOR KIC {kic_id}")
        print(f"="*40)
        print(f"SEISMIC:")
        # Seismic Print Logic
        v_max_str = f"{v_max_lit:7.2f}" if v_max_lit is not None else "    N/A"
        d_nu_str = f"{d_nu_lit:7.2f}" if d_nu_lit is not None else "    N/A"
        
        print(f"  v_max: Catalog = {v_max_str} | Calc = {our_v_max:7.2f}")
        print(f"  D_nu:  Catalog = {d_nu_str} | Calc = {our_d_nu:7.2f}")
        
        # Physical/Chemical Print Logic
        mass_str = f"{mass_lit:7.2f}" if mass_lit is not None else "    N/A"
        feh_str = f"{feh_lit:7.2f}" if feh_lit is not None else "    N/A"
        
        print(f"PHYSICAL/CHEMICAL:")
        print(f"  Mass:  {mass_str} M_sun")
        print(f"  [Fe/H]: {feh_str} (Metallicity)")


        # --- RIGOROUS SEISMIC ESTIMATION BLOCK ---
        from scipy.optimize import curve_fit

        # 1. Precise v_max via Gaussian Fitting
        # We fit a Gaussian to the 'hump_envelope' we already calculated
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        #variables equal tp later ones
        zoom_mask1 = (p_freq > our_v_max - (our_d_nu*6)) & (p_freq < our_v_max + (our_d_nu*6))
        f_zoom1 = p_freq[zoom_mask1]
        p_zoom1 = p_power[zoom_mask1]
        bg_smooth1 = gaussian_filter1d(p_zoom1, sigma=100)
        osc_hump1 = p_zoom1 - bg_smooth1
        # Apply a gentler smoothing to show the Gaussian-like envelope
        hump_envelope1 = gaussian_filter1d(osc_hump1, sigma=30)
        try:
            # Initial guess: [amplitude, center (from simple search), width (guess 0.1*v_max)]
            popt, _ = curve_fit(gaussian, f_zoom1, hump_envelope1, 
                               p0=[np.max(hump_envelope1), our_v_max, our_v_max*0.1])
            precise_v_max = popt[1]
        except:
            precise_v_max = our_v_max # Fallback

        # 2. Precise D_nu via Autocorrelation Function (ACF)
        # This measures the actual spacing of the peaks instead of using a formula
        def calculate_precise_dnu(freq, power, central_v_max):
            # Focus on the region immediately around v_max
            mask = (freq > central_v_max * 0.7) & (freq < central_v_max * 1.3)
            f_acf = freq[mask]
            p_acf = power[mask]
            
            # Compute Autocorrelation
            n = len(p_acf)
            acf = np.correlate(p_acf - np.mean(p_acf), p_acf - np.mean(p_acf), mode='full')[n-1:]
            lags = np.arange(len(acf)) * (f_acf[1] - f_acf[0])
            
            # Search for the D_nu peak in the ACF
            # Expected D_nu is roughly our_d_nu; we search in a window around it
            expected_dnu = 0.22 * (central_v_max**0.77)
            acf_mask = (lags > expected_dnu * 0.8) & (lags < expected_dnu * 1.2)
            
            if np.any(acf_mask):
                precise_dnu = lags[acf_mask][np.argmax(acf[acf_mask])]
                return precise_dnu
            return expected_dnu

        precise_d_nu = calculate_precise_dnu(p_freq, p_power, precise_v_max)

        # Update the Printout for comparison
        print(f"PRECISE ESTIMATES (Curve Fitting & ACF):")
        print(f"  v_max: {precise_v_max:7.2f} µHz")
        print(f"  D_nu:  {precise_d_nu:7.2f} µHz")
        print("-" * 40)
        # --- END OF RIGOROUS BLOCK ---

        ax4.loglog(p_freq, p_power, color='royalblue', lw=0.8)
        ax4.axvline(our_v_max, color='green', lw=2, label=f'Calc v_max: {our_v_max:.1f}')
        ax4.axvline(precise_v_max, color='blue', lw=2, label=f'Precise v_max: {precise_v_max:.1f}')
        if v_max_lit:
            ax4.axvline(v_max_lit, color='red', linestyle='--', alpha=0.6, label=f'Lit v_max: {v_max_lit:.1f}')
        ax4.set_xlim(0.01, 5000); ax4.set_xlabel("Frequency (µHz)")
        ax4.legend()
        



        # Secondary Figure: Linear Zoom with Envelope
        fig_z, (az1, az2, az3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        plt.subplots_adjust(hspace=0.3) # Spacing for the zoom figure
        
        zoom_mask = (p_freq > our_v_max - (our_d_nu*6)) & (p_freq < our_v_max + (our_d_nu*6))
        f_zoom = p_freq[zoom_mask]
        p_zoom = p_power[zoom_mask]
        
        # PANEL A: Raw PSD + Highly Smoothed Background (Yellow curve equivalent)
        az1.plot(f_zoom, p_zoom, color='black', lw=0.5, alpha=0.6, label='Raw PSD')
        # Background estimation (Harvey-like smoothing)
        bg_smooth = gaussian_filter1d(p_zoom, sigma=100) 
        az1.plot(f_zoom, bg_smooth, color='gold', lw=2.5, label='Stellar Background (Smoothed)')
        az1.set_title(f"KIC {kic_id} - Total Power Density")
        az1.set_ylabel("ppm²/µHz")
        az1.legend(loc='upper right')

        # PANEL B: The Michel Hump (Isolated Oscillation Power - Blue curve equivalent)
        # We subtract the background to isolate the power excess
        osc_hump = p_zoom - bg_smooth
        # Apply a gentler smoothing to show the Gaussian-like envelope
        hump_envelope = gaussian_filter1d(osc_hump, sigma=30)
        
        az2.plot(f_zoom, osc_hump, color='gray', lw=0.5, alpha=0.5)
        az2.plot(f_zoom, hump_envelope, color='royalblue', lw=2, label='Isolated Oscillation Hump')
        az2.fill_between(f_zoom, 0, hump_envelope, color='royalblue', alpha=0.2)
        az2.set_title("Oscillation Power Contribution Alone")
        az2.set_ylabel("ppm²/µHz")
        az2.legend(loc='upper right')

        # PANEL C: Flattened View with D_nu Grid
        p_flat = p_zoom * (f_zoom**2)
        az3.plot(f_zoom, p_flat, color='darkorange', lw=0.8)
        for i in range(-5, 6):
            az3.axvline(precise_v_max + (i*precise_d_nu), color='black', alpha=0.2, ls='--')
        
        az3.set_title(f"Flattened Mode Spacing (Δν ≈ {precise_d_nu:.2f} µHz)")
        az3.set_xlabel("Frequency (µHz)")
        az3.set_ylabel("Weighted Power")
        
    plt.tight_layout()
    plt.show()

# Run for a Red Giant from APOKASC-3
plot_comprehensive_diagnostic(1163114)
#some stars: 1027337