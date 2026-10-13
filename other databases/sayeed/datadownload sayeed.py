import pandas as pd
from astroquery.mast import Observations
import time
import os

# 1. Setup
base_download_dir = "./kepseismic_data"
file_path = "./sayeed_catalogue.txt"

# 2. Load Catalog (New, Robust Version)
# We use sep='\s+' to tell pandas "any amount of whitespace is a separator"
# This handles the 7-digit vs 8-digit shift automatically.
df = pd.read_csv(file_path, skiprows=45, header=None, sep='\s+', engine='python')

# The KIC ID is in the first column (index 0)
kic_ids = df[0].tolist()



# --- ADDED: Stats Counters ---
stats = {"complete": 0, "incomplete": 0, "no_data": 0}

# 3. The Loop
for i, kic in enumerate(kic_ids):
    kic_padded = str(int(kic)).zfill(9)
    print(f"[{i+1}/{len(kic_ids)}] Processing KIC {kic}...")

    # --- ADDED: Quick Check to skip already downloaded stars ---
    # This looks for the directory MAST creates for this specific KIC
    is_already_done = False
    for root, dirs, files in os.walk(base_download_dir):
        if f"kplr{kic_padded}" in root:
            fits_count = len([f for f in files if f.endswith('.fits')])
            if fits_count >= 6:
                print(f"   Already complete (Found {fits_count} files). Skipping.")
                stats["complete"] += 1
                is_already_done = True
                break
    
    if is_already_done: continue

    try:
        # 1. Format the KIC ID exactly as MAST stores it (e.g., kplr000003270)
        # zfill(9) ensures we have 9 digits with leading zeros
        target_name = f"kplr{str(int(kic)).zfill(9)}"
        
        # 2. Search MAST directly by Target Name and Provenance
        # This skips the coordinate resolver (no "sky position" needed)
        obs = Observations.query_criteria(target_name=target_name, 
                                         provenance_name='KEPSEISMIC')
        
        if len(obs) > 0:
            # 3. Get products and filter for FITS
            products = Observations.get_product_list(obs)
            final_products = Observations.filter_products(products, extension="fits")
            
            if len(final_products) > 0:
                print(f"   Found {len(final_products)} files for {target_name}. Downloading...")
                Observations.download_products(final_products, download_dir=base_download_dir)
            else:
                print(f"   No FITS found for {target_name}")
        else:
            print(f"   No KEPSEISMIC data found for {target_name} in MAST database.")

        time.sleep(0.1) # Faster search, so we can reduce the sleep time

    except Exception as e:
        print(f"!!! Error with KIC {kic}: {e}")
        continue

# --- ADDED: Final Summary Report ---
print("\n" + "="*30)
print("DOWNLOAD SUMMARY")
print("="*30)
print(f"Stars with 6+ files: {stats['complete']}")
print(f"Stars with <6 files: {stats['incomplete']}")
print(f"Stars with no data:  {stats['no_data']}")
print("="*30)