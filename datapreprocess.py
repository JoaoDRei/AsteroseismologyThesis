import os
import numpy as np
import lightkurve as lk
from tqdm import tqdm
from astropy.stats import sigma_clip
import warnings

# Silencing the warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

raw_data_root = "./kepseismic_data/mastDownload/HLSP"
processed_dir = "./processed_ml_data"
os.makedirs(processed_dir, exist_ok=True)

def process_stars(filter_type="80d"):
    star_folders = [f for f in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, f))]
    
    # tqdm is only on the outer loop
    for folder in tqdm(star_folders, desc=f"Processing {filter_type}"):
        folder_path = os.path.join(raw_data_root, folder)
        fits_files = [f for f in os.listdir(folder_path) if filter_type in f and 'cor-filt-inp.fits' in f]
        
        for fits_name in fits_files:
            file_path = os.path.join(folder_path, fits_name)
            try:
                # 1. Load and Remove NaNs
                lc = lk.read(file_path).remove_nans()
                
                # 2. Sigma Clip to remove those huge spikes (5-sigma)
                clipped = sigma_clip(lc.flux.value, sigma=5, cenfunc='median')
                mask = ~clipped.mask
                
                clean_time = lc.time.value[mask]
                clean_flux_raw = lc.flux.value[mask]
                
                # 3. Standardize (Mean=0, Std=1)
                f_mean = np.nanmean(clean_flux_raw)
                f_std = np.nanstd(clean_flux_raw)
                if f_std == 0: continue
                
                flux_scaled = (clean_flux_raw - f_mean) / f_std
                
                # 4. Save Time Domain NPY
                kic_id = fits_name.split('_')[4].split('-')[0]
                lc_path = os.path.join(processed_dir, f"{kic_id}_{filter_type}_clean.npy")
                np.save(lc_path, np.array([clean_time, flux_scaled], dtype=np.float32))
                
                # 5. Generate and Save PSD NPY
                psd_path = os.path.join(processed_dir, f"{kic_id}_{filter_type}_psd.npy")
                temp_lc = lk.LightCurve(time=clean_time, flux=flux_scaled)
                pg = temp_lc.to_periodogram(method='lombscargle', normalization='psd')
                
                # Filtering frequencies to keep the file size small (Standard Kepler range)
                pg = pg[(pg.frequency.value >= 0.01) & (pg.frequency.value <= 5000)]
                np.save(psd_path, np.array([pg.frequency.value, pg.power.value], dtype=np.float32))
                    
            except Exception:
                # Use tqdm.write if you absolutely need to see the error
                # tqdm.write(f"Skipping {fits_name}")
                continue

if __name__ == "__main__":
    for f in ["80d", "55d", "20d"]:
        process_stars(f)