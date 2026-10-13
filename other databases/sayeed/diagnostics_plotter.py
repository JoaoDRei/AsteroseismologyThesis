import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd

def plot_comprehensive_diagnostic(kic_id, base_dir="./kepseismic_data", proc_dir="./processed_ml_data", filter_type="80d"):
    kic_padded = str(int(kic_id)).zfill(9)

    # --- ADD: Load Official Catalog Values ---
    
    v_max_lit, d_nu_lit = None, None
    try:
        df_cat = pd.read_csv("./sayeed_catalogue.txt", skiprows=45, header=None, sep='\s+', engine='python')
        star_data = df_cat[df_cat[0] == int(kic_id)]
        if not star_data.empty:
            v_max_lit = star_data.iloc[0, 7]  # numax column
            d_nu_lit = star_data.iloc[0, 9]   # Dnu column
    except Exception as e:
        print(f"Catalog Load Error: {e}")

    # 1. Find the Raw FITS
    fits_pattern = os.path.join(base_dir, "**", f"*{kic_padded}*{filter_type}*filt-inp.fits")
    fits_files = glob.glob(fits_pattern, recursive=True)
    
    # 2. Find the Processed NPY Files
    # (Assuming we save two separate NPYs or one for each domain)
    lc_npy_path = os.path.join(proc_dir, f"kplr{kic_padded}_{filter_type}_clean.npy")
    psd_npy_path = os.path.join(proc_dir, f"kplr{kic_padded}_{filter_type}_psd.npy")

    # Setup 4-panel plot (Vertical stack like yours)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

    # --- PANEL 1: RAW FITS (Time Domain) ---
    if fits_files:
        lc = lk.read(fits_files[0])
        lc.plot(ax=ax1, color='gray', lw=0.5, label='Raw FITS (PPM)')
        ax1.set_title(f"KIC {kic_id} - Original KEPSEISMIC Data")
    else:
        ax1.text(0.5, 0.5, "FITS Not Found", ha='center')

    # --- PANEL 2: PROCESSED NPY (ML Input - Time Domain) ---
    if os.path.exists(lc_npy_path):
        data_lc = np.load(lc_npy_path)
        p_time, p_flux = data_lc[0], data_lc[1]
        ax2.plot(p_time, p_flux, color='black', lw=0.5, label='Standardized NPY')
        ax2.set_title(f"Processed ML Input (Time Domain)")
        ax2.set_ylabel("Standardized Flux")
    else:
        ax2.text(0.5, 0.5, "LC NPY Not Found", ha='center')

    # --- PANEL 3: RAW PSD (Scientific Check) ---
    if fits_files:
        pg = lc.to_periodogram(method='lombscargle', normalization='psd')
        pg.plot(ax=ax3, color='gray', lw=0.8, alpha=0.5, label='Raw PSD')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlim(0.01, 5000)
        ax3.set_title("Raw Power Spectrum (From FITS)")
    
    # --- PANEL 4: PROCESSED PSD (From NPY) ---
    if os.path.exists(psd_npy_path):
        data_psd = np.load(psd_npy_path)
        p_freq, p_power = data_psd[0], data_psd[1]
        
        # --- ADD: Calculate Our Values ---
       # --- FIXED: Weighting the power to find the high-frequency hump ---
        from scipy.ndimage import gaussian_filter1d
        
        # 1. Ignore the extreme low-end
        search_mask = (p_freq > 20.0) 
        f_search = p_freq[search_mask]
        p_search = p_power[search_mask]

        # 2. WEIGHTING: Multiply power by frequency squared 
        # This cancels the 1/v^2 granulation decay and 'flattens' the slope
        weighted_p = p_search * (f_search**2)

        # 3. Smooth the WEIGHTED data to find the hump center
        smoothed_p = gaussian_filter1d(weighted_p, sigma=30) 
        our_v_max = f_search[np.argmax(smoothed_p)]
        
        # 4. Estimate Dnu
        our_d_nu = 0.22 * (our_v_max**0.77)

        # Terminal Comparison
        print(f"\n--- KIC {kic_id} ---")
        if v_max_lit:
            print(f"Catalog v_max: {v_max_lit:.2f} | Our v_max: {our_v_max:.2f}")
        else:
            print(f"Our Calculated v_max: {our_v_max:.2f}")
        print("--------------------\n")

        ax4.loglog(p_freq, p_power, color='royalblue', lw=0.8, label='Processed PSD')
        ax4.set_title("Processed Power Spectrum (From NPY)")
        ax4.set_xlim(0.01, 5000)
        ax4.set_xlabel("Frequency (µHz)")
        ax4.axvline(our_v_max, color='green', linestyle='-', lw=2, label=f'Our v_max: {our_v_max:.1f}')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "PSD NPY Not Found", ha='center')

    plt.tight_layout()

    # --- ADD: Separate Zoom Window ---
    if os.path.exists(psd_npy_path):
        plt.figure(figsize=(10, 5))
        # Use our calculated v_max to set the zoom window
        z_min, z_max = our_v_max - (4 * our_d_nu), our_v_max + (4 * our_d_nu)
        mask = (p_freq > z_min) & (p_freq < z_max)
        
        plt.plot(p_freq[mask], p_power[mask], color='royalblue', lw=1, label='Processed PSD')
        plt.axvline(our_v_max, color='green', lw=2, label=f'Our v_max ({our_v_max:.1f})')
        
        # Draw Delta-Nu Ladder (visual check of peak spacing)
        for i in range(-3, 4):
            plt.axvline(our_v_max + (i * our_d_nu), color='black', linestyle=':', alpha=0.3)
            
        plt.title(f"KIC {kic_id} - Oscillation Zoom (Linear Scale)")
        plt.xlabel("Frequency (µHz)")
        plt.ylabel("Power")
        plt.legend()
    plt.show()

# Run it!
plot_comprehensive_diagnostic(12554802, filter_type="80d")