import pandas as pd
from astroquery.mast import Observations
from astropy.table import Table
import time
import os

# 1. Setup - Fresh folder for DR25 Short Cadence
base_download_dir = "./kepler_dr25_sc"
file_path = "./sayeed_catalogue.txt"
os.makedirs(base_download_dir, exist_ok=True)

# 2. Load Catalog
df = pd.read_csv(file_path, skiprows=45, header=None, sep='\s+', engine='python')
kic_ids = df[0].tolist()

# Stats Counters
stats = {"complete": 0, "downloaded": 0, "no_sc": 0}

# 3. The Loop
for i, kic in enumerate(kic_ids):
    kic_padded = str(int(kic)).zfill(9)
    target_name = f"kplr{kic_padded}"
    print(f"[{i+1}/{len(kic_ids)}] Processing {target_name}...")

    # --- Robust Check: Skip if already downloaded ---
    is_already_done = False
    for root, dirs, files in os.walk(base_download_dir):
        if target_name in root:
            # Look for Short Cadence fits files specifically
            sc_fits = [f for f in files if 'slc.fits' in f]
            if len(sc_fits) > 0:
                print(f"   Already complete (Found {len(sc_fits)} SC files). Skipping.")
                stats["complete"] += 1
                is_already_done = True
                break
    
    if is_already_done: continue

    try:
        # 1. Search Official Kepler Mission (DR25)
        obs = Observations.query_criteria(target_name=target_name, 
                                          project='Kepler',
                                          obs_collection='Kepler')
        
        if len(obs) > 0:
            # 2. Get Product List
            products = Observations.get_product_list(obs)
            
            # 3. Filter for 'slc' (Short Light Cadence) and 'fits'
            # We filter the Table directly to keep it as a Table object
            mask = [('slc.fits' in row['productFilename']) for row in products]
            sc_table = products[mask]
            
            if len(sc_table) > 0:
                print(f"   Found {len(sc_table)} SC files. Downloading...")
                # Observations.download_products works best with the filtered Table
                Observations.download_products(sc_table, download_dir=base_download_dir)
                stats["downloaded"] += 1
            else:
                print(f"   No Short Cadence found for {target_name}")
                stats["no_sc"] += 1
        else:
            print(f"   No Kepler data found for {target_name}")
            stats["no_sc"] += 1

        time.sleep(0.1) 

    except Exception as e:
        print(f"!!! Error with KIC {kic}: {e}")
        continue

# --- Final Summary ---
print("\n" + "="*30)
print("DR25 SC DOWNLOAD SUMMARY")
print("="*30)
print(f"Stars already done:    {stats['complete']}")
print(f"New stars downloaded:  {stats['downloaded']}")
print(f"Stars with no SC:      {stats['no_sc']}")
print("="*30)

#==============================
#DR25 SC DOWNLOAD SUMMARY
#==============================
#Stars already done:    168
#New stars downloaded:  594
#Stars with no SC:      2
#==============================