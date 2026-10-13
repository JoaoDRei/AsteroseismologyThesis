import os
import numpy as np
import lightkurve as lk
import pandas as pd
from tqdm import tqdm
from astropy.stats import sigma_clip
import warnings
import glob

# Silencing warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
# This path matches where your download script saves the data
raw_data_root = "./kepseismic_data_APOKASC3/mastDownload/HLSP"
processed_dir = "./processed_ml_data_APOKASC3"
catalog_path = "./table4APOKASC.txt"

os.makedirs(processed_dir, exist_ok=True)

def load_catalog_ids():
    """Loads KIC IDs from the APOKASC-3 catalog."""
    specs = [(0, 8)]
    names = ['KIC']
    df = pd.read_fwf(catalog_path, colspecs=specs, names=names, skiprows=111)
    # Convert to set for O(1) lookup speed
    return set(df['KIC'].dropna().astype(int).tolist())

def process_apokasc_stars(filter_type="80d"):
    valid_kics = load_catalog_ids()
    processed_count = 0
    
    # RECURSIVE SEARCH: This finds all fits files with the filter name anywhere inside raw_data_root
    search_pattern = os.path.join(raw_data_root, "**", f"*{filter_type}*cor-filt-inp.fits")
    fits_files = glob.glob(search_pattern, recursive=True)
    
    if not fits_files:
        print(f"No files found for {filter_type} in {raw_data_root}. Check your download path!")
        return

    for file_path in tqdm(fits_files, desc=f"Processing {filter_type}"):
        # Extract KIC from the filename (usually contains kplrXXXXXXXXX)
        filename = os.path.basename(file_path)
        try:
            # Finds 'kplr' and takes the 9 digits following it
            k_idx = filename.find('kplr')
            if k_idx == -1: continue
            kic_id = int(filename[k_idx+4 : k_idx+13])
        except (ValueError, IndexError):
            continue
            
        # Only process if it's in our catalog whitelist
        if kic_id not in valid_kics:
            continue
            
        try:
            # 1. Load and Remove NaNs
            lc = lk.read(file_path).remove_nans()
            if len(lc) == 0: continue
            
            # 2. Sigma Clipping (5-sigma) #removes outliers
            clipped = sigma_clip(lc.flux.value, sigma=5, cenfunc='median')
            mask = ~clipped.mask
            
            clean_time = lc.time.value[mask]
            clean_flux_raw = lc.flux.value[mask]
            
            # 3. Standardization (Mean=0, Std=1)
            f_mean = np.nanmean(clean_flux_raw)
            f_std = np.nanstd(clean_flux_raw)
            if f_std == 0: continue
            
            flux_scaled = (clean_flux_raw - f_mean) / f_std #standardization
            
            # 4. Save Light Curve NPY
            kic_str = str(kic_id).zfill(9)
            lc_path = os.path.join(processed_dir, f"kplr{kic_str}_{filter_type}_clean.npy")
            np.save(lc_path, np.array([clean_time, flux_scaled], dtype=np.float32))
            
            # 5. Save PSD NPY
            psd_path = os.path.join(processed_dir, f"kplr{kic_str}_{filter_type}_psd.npy")
            temp_lc = lk.LightCurve(time=clean_time, flux=flux_scaled)
            pg = temp_lc.to_periodogram(method='lombscargle', normalization='psd')
            
            
            pg = pg[(pg.frequency.value >= 0.01) & (pg.frequency.value <= 5000)]
            np.save(psd_path, np.array([pg.frequency.value, pg.power.value], dtype=np.float32))
            
            processed_count += 1
                
        except Exception as e:
            # Uncomment for debugging specific file errors:
            # print(f"Error processing {filename}: {e}")
            continue

    print(f"\nSuccessfully processed {processed_count} files for {filter_type}.")

if __name__ == "__main__":
    # Prioritize 80d, then try others
    for f in ["80d", "55d", "20d"]:
        process_apokasc_stars(f)