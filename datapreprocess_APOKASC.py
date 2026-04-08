import os
import numpy as np
import lightkurve as lk
import pandas as pd
from tqdm import tqdm
from astropy.stats import sigma_clip
import warnings

# Silencing warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
raw_data_root = "./kepseismic_data_APOKASC3/mastDownload/HLSP"
processed_dir = "./processed_ml_data_APOKASC3"
catalog_path = "./table4APOKASC.txt"
os.makedirs(processed_dir, exist_ok=True)

def load_catalog_ids():
    """Loads KIC IDs from the APOKASC-3 catalog to use as a whitelist."""
    # Using the fixed-width spec for KIC (Doc: 1-8)
    specs = [(0, 8)]
    names = ['KIC']
    df = pd.read_fwf(catalog_path, colspecs=specs, names=names, skiprows=111)
    return set(df['KIC'].astype(int).tolist())

def process_apokasc_stars(filter_type="80d"):
    # Load the whitelist of IDs we actually care about
    valid_kics = load_catalog_ids()
    
    # Get all KIC-specific folders in the MAST directory
    star_folders = [f for f in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, f))]
    
    processed_count = 0
    
    for folder in tqdm(star_folders, desc=f"Preprocessing {filter_type}"):
        # Extract KIC from folder name (usually kplrXXXXXXXXX)
        try:
            folder_kic = int(folder.replace('kplr', ''))
        except ValueError:
            continue
            
        # Only process if it's in our APOKASC-3 catalog
        if folder_kic not in valid_kics:
            continue
            
        folder_path = os.path.join(raw_data_root, folder)
        # Look for the corrected-filtered-interpolated KEPSEISMIC files
        fits_files = [f for f in os.listdir(folder_path) if filter_type in f and 'cor-filt-inp.fits' in f]
        
        for fits_name in fits_files:
            file_path = os.path.join(folder_path, fits_name)
            
            try:
                # 1. Load and Remove NaNs
                lc = lk.read(file_path).remove_nans()
                
                # 2. Sigma Clipping (5-sigma)
                # Removes non-physical spikes that could confuse the Transformer's attention layers
                clipped = sigma_clip(lc.flux.value, sigma=5, cenfunc='median')
                mask = ~clipped.mask
                
                clean_time = lc.time.value[mask]
                clean_flux_raw = lc.flux.value[mask]
                
                # 3. Standardization (Mean=0, Std=1)
                # CRITICAL for Neural Networks to converge quickly
                f_mean = np.nanmean(clean_flux_raw)
                f_std = np.nanstd(clean_flux_raw)
                if f_std == 0: continue
                
                flux_scaled = (clean_flux_raw - f_mean) / f_std
                
                # 4. Save Time Domain NPY (Input for Astroconformer)
                kic_str = str(folder_kic).zfill(9)
                lc_path = os.path.join(processed_dir, f"kplr{kic_str}_{filter_type}_clean.npy")
                np.save(lc_path, np.array([clean_time, flux_scaled], dtype=np.float32))
                
                # 5. Save PSD NPY (For traditional physics / sanity checks)
                psd_path = os.path.join(processed_dir, f"kplr{kic_str}_{filter_type}_psd.npy")
                temp_lc = lk.LightCurve(time=clean_time, flux=flux_scaled)
                pg = temp_lc.to_periodogram(method='lombscargle', normalization='psd')
                
                # Restrict range to reduce file size (Red Giants live in 1-1000 uHz)
                pg = pg[(pg.frequency.value >= 0.1) & (pg.frequency.value <= 3000)]
                np.save(psd_path, np.array([pg.frequency.value, pg.power.value], dtype=np.float32))
                
                processed_count += 1
                    
            except Exception:
                continue

    print(f"\nSuccessfully processed {processed_count} files for {filter_type}.")

if __name__ == "__main__":
    # Start with 80d as it's the most useful for Red Giant oscillations
    for f in ["80d", "55d", "20d"]:
        process_apokasc_stars(f)