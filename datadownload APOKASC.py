import pandas as pd
from astroquery.mast import Observations
import time
import os

# --- 1. Setup ---
base_download_dir = "./kepseismic_data_APOKASC3"
# Ensure the file name matches your downloaded APOKASC-3 file
file_path = "./table4APOKASC.txt" 

# --- 2. Load and Filter APOKASC-3 Catalog ---
# Using fixed-width (fwf) because the metadata shows specific character ranges
# Columns: 1-8 (KIC), 23-30 (Category/CatTab), 50-59 (Numax), 72-81 (DNu), 127-136 (Mass), 149-158 (Radius), 237-246 ([Fe/H]), 259-268 ([a/Fe])
# Conversion applied: (DocStart - 1, DocEnd)
specs = [
    (0, 8),     # KIC (Doc: 1-8)
    (9, 16),    # EvolState (Doc: 10-16)
    (22, 30),   # CatTab (Doc: 23-30) -> This identifies 'Gold'
    (49, 59),   # Numax (Doc: 50-59)
    (71, 81),   # DNu (Doc: 72-81)
    (126, 136), # Mass (Doc: 127-136)
    (148, 158), # Radius (Doc: 149-158)
    (236, 246), # [Fe/H] (Doc: 237-246)
    (258, 268)  # [a/Fe] (Doc: 259-268)
]

names = ['KIC', 'EvolState', 'CatTab', 'Numax', 'DNu', 'Mass', 'Radius', 'FeH', 'Alpha']

# Now read the file
print(f"Reading {file_path}...")
df = pd.read_fwf(file_path, colspecs=specs, names=names, skiprows=111)

# Filter for 'Gold' sample only
# Note: strip() is used because fixed-width often leaves trailing spaces
df['CatTab'] = df['CatTab'].astype(str).str.strip()
gold_df = df[df['CatTab'] == 'Gold'].copy()

kic_ids = gold_df['KIC'].tolist()
print(f"Found {len(kic_ids)} 'Gold' stars in the catalog. Starting download...")

# --- 3. The Download Loop ---
stats = {"complete": 0, "downloaded": 0, "no_data": 0}


# Change this number to the KIC ID where you want to resume
start_kic =  10286378 

# 8025383 problems started at this one

# Find where that KIC is in the list and slice it
start_index = kic_ids.index(start_kic)
#start_index=0 #start from the beginning
for i, kic in enumerate(kic_ids[start_index:], start=start_index):
    # Skip potential headers or malformed rows
    try:
        kic_val = int(kic)
    except:
        continue

    kic_padded = str(kic_val).zfill(9)
    target_name = f"kplr{kic_padded}"
    
    print(f"[{i+1}/{len(kic_ids)}] Processing KIC {kic_val}...")

    # --- Quick Check: Skip already downloaded stars ---
    is_already_done = False
    if os.path.exists(base_download_dir):
        for root, dirs, files in os.walk(base_download_dir):
            if target_name in root:
                fits_count = len([f for f in files if f.endswith('.fits')])
                if fits_count >= 1: # Adjust this number if you expect a specific count
                    print(f"    Already exists (Found {fits_count} files). Skipping.")
                    stats["complete"] += 1
                    is_already_done = True
                    break
    
    if is_already_done: continue

    try:
        # Search MAST for KEPSEISMIC provenance
        obs = Observations.query_criteria(target_name=target_name, 
                                          provenance_name='KEPSEISMIC')
        
        if len(obs) > 0:
            products = Observations.get_product_list(obs)
            final_products = Observations.filter_products(products, extension="fits")
            
            if len(final_products) > 0:
                print(f"    Found {len(final_products)} KEPSEISMIC files. Downloading...")
                Observations.download_products(final_products, download_dir=base_download_dir)
                stats["downloaded"] += 1
            else:
                print(f"    No FITS found for {target_name}")
                stats["no_data"] += 1
        else:
            print(f"    No KEPSEISMIC data found for {target_name}")
            stats["no_data"] += 1

        time.sleep(0.1) 

    except Exception as e:
        print(f"!!! Error with KIC {kic_val}: {e}")
        continue

# --- Final Summary Report ---
print("\n" + "="*30)
print("APOKASC-3 GOLD DOWNLOAD SUMMARY")
print("="*30)
print(f"Previously completed: {stats['complete']}")
print(f"Newly downloaded:     {stats['downloaded']}")
print(f"No data found:        {stats['no_data']}")
print("="*30)