import os
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import glob

# --- Settings ---
input_dir = "./kepler_dr25_sc"
output_dir = "./processed_raw_sc"
os.makedirs(output_dir, exist_ok=True)

def preprocess_star(kic_id):
    # 1. Find all 'slc' files for this KIC in the MAST subfolder structure
    search_path = os.path.join(input_dir, f"**/kplr{kic_id.zfill(9)}*slc.fits")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        return None

    # Sort files by name (which includes the timestamp) to keep them chronological
    files.sort()
    
    all_time = []
    all_flux = []

    for f in files:
        with fits.open(f) as hdul:
            data = hdul[1].data
            # PDCSAP_FLUX is the 'Pre-search Data Conditioning' flux (cleaned by NASA)
            time = data['TIME']
            flux = data['PDCSAP_FLUX']
            
            # Remove NaNs immediately
            mask = ~np.isnan(flux)
            if len(time[mask]) > 0:
                # Normalize each quarter individually to 1.0 to remove jumps
                q_flux = flux[mask] / np.nanmedian(flux[mask])
                all_time.append(time[mask])
                all_flux.append(q_flux)

    if not all_flux:
        return None

    # Concatenate all quarters
    full_time = np.concatenate(all_time)
    full_flux = np.concatenate(all_flux)

    # 2. High-pass Filter (Remove slow instrumental drifts)
    # Using a large window (e.g., 2 days) to keep seismic signals (< 1 hour) safe
    # Short Cadence is ~1 min, so 2 days is ~2880 points.
    window_size = 2881 
    trend = savgol_filter(full_flux, window_size, polyorder=1)
    clean_flux = full_flux - trend # This centers the data at 0.0

    return clean_flux

# --- Execution Loop ---
# Get list of KIC IDs from the downloaded folders
subdirs = [d for d in os.listdir(os.path.join(input_dir, "mastDownload", "Kepler"))]
kic_list = [d.replace("kplr", "").split("_")[0] for d in subdirs]

print(f"Starting preprocessing for {len(kic_list)} stars...")

for kic in kic_list:
    try:
        flux_array = preprocess_star(kic)
        if flux_array is not None:
            np.save(os.path.join(output_dir, f"{kic}_raw.npy"), flux_array)
            print(f"Successfully processed KIC {kic} | Length: {len(flux_array)} pts")
    except Exception as e:
        print(f"Error processing KIC {kic}: {e}")

print("\nPre-processing complete. Files saved in:", output_dir)