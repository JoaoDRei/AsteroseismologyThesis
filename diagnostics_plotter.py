import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_comprehensive_diagnostic(kic_id, base_dir="./kepseismic_data", proc_dir="./processed_ml_data", filter_type="80d"):
    kic_padded = str(int(kic_id)).zfill(9)
    
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
        ax4.loglog(p_freq, p_power, color='royalblue', lw=0.8, label='Processed PSD')
        ax4.set_title("Processed Power Spectrum (From NPY)")
        ax4.set_xlim(0.01, 5000)
        ax4.set_xlabel("Frequency (µHz)")
    else:
        ax4.text(0.5, 0.5, "PSD NPY Not Found", ha='center')

    plt.tight_layout()
    plt.show()

# Run it!
plot_comprehensive_diagnostic(4351319, filter_type="80d")