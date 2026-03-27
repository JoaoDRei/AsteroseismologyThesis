import os
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from tqdm import tqdm # Install this for a nice progress bar: pip install tqdm

# 1. Setup Paths
raw_data_root = "./kepseismic_data/mastDownload/HLSP"
processed_dir = "./processed_ml_data"
os.makedirs(processed_dir, exist_ok=True)

def process_stars():
    # Find all subdirectories in the HLSP folder
    star_folders = [f for f in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, f))]
    
    print(f"Found {len(star_folders)} potential data folders.")

    for folder in tqdm(star_folders):
        folder_path = os.path.join(raw_data_root, folder)
        
        # We focus on the '80d' Light Curves (LC) as they are best for granulation
        # Filename example: hlsp_kepseismic_kepler_phot_kplr002162635-80d_kepler_v1_cor-filt-inp.fits
        fits_files = [f for f in os.listdir(folder_path) if '80d' in f and 'cor-filt-inp.fits' in f]
        
        for fits_name in fits_files:
            file_path = os.path.join(folder_path, fits_name)
            
            try:
                # 2. Use Lightkurve to read the FITS
                lc = lk.read(file_path)
                
                # Clean the data: remove NaNs and normalize
                lc = lc.remove_nans().normalize()
                
                # Extract the 1D vectors
                time = lc.time.value
                flux = lc.flux.value
                
                # 3. Save as a compact NumPy file
                # We name it by the KIC ID found in the filename
                kic_id = fits_name.split('_')[4].split('-')[0] # Extracts 'kplr002162635'
                save_path = os.path.join(processed_dir, f"{kic_id}_80d_clean.npy")
                
                # We save time and flux together in one file
                np.save(save_path, np.array([time, flux]))
                
                # --- VISUALIZATION (Only for the first star to check) ---
                if folder == star_folders[0]:
                    plt.figure(figsize=(12, 4))
                    plt.plot(time, flux, color='black', lw=0.5)
                    plt.title(f"Granulation Signal: {kic_id} (80d Filtered)")
                    plt.xlabel("Time (Days)")
                    plt.ylabel("Normalized Flux")
                    plt.show()
                    
            except Exception as e:
                print(f"Error processing {fits_name}: {e}")

if __name__ == "__main__":
    process_stars()