import pandas as pd
import glob
import os
import numpy as np

def create_manifest(processed_dir, catalog_path,hon_path, general_catalog_path, output_csv="./Baseline/manifest.csv"):
    # 1. Load the APOKASC catalog
    # Adjust names/colspecs based on your specific table4APOKASC.txt format
    names = ['KIC', 'EvolState', 'nu_max', 'delta_nu', 'Teff']
    colspecs = [(0, 8), (9,16),(49, 59), (71, 81), (192,202)] # positions for KIC and nu_max
    catalog = pd.read_fwf(catalog_path, colspecs=colspecs, names=names, skiprows=111)
    catalog = catalog.dropna(subset=['nu_max']) # Remove entries without nu_max
    catalog['KIC'] = catalog['KIC'].astype(int) # Ensure KIC is integer for matching

    #Load the Hon 2019 table:
    hon_data = pd.read_csv(hon_path, sep='|') # Adjust separator if needed
    hon_data = hon_data[['KIC', 'nu_max']].rename(columns={'nu_max': 'nu_max_hon'})   
    hon_data = hon_data[hon_data['nu_max_hon'] > 0]

    #Load the General APOKASC catalogue with all the pipelines and get the SYD, A2Z and DIA pipelines
    namesGEN=['KIC', 'nu_max_syd', 'delta_nu_syd', 'nu_max_a2z', 'delta_nu_a2z', 'nu_max_dia', 'delta_nu_dia'] 
    colspecsGEN = [(0, 8), (92,102),(216, 226), (103, 113), (227,237), (114,124), (238,248)] 
    general_catalog=pd.read_fwf(general_catalog_path, colspecs=colspecsGEN, names=namesGEN, skiprows=91)
    general_catalog.replace(-9999, np.nan, inplace=True)
    general_catalog=general_catalog.dropna(subset=['nu_max_syd'])
    general_catalog=general_catalog.dropna(subset=['nu_max_a2z'])
    general_catalog=general_catalog.dropna(subset=['nu_max_dia'])
    general_catalog['KIC'] = general_catalog['KIC'].astype(int)

    # 3. Find all light curveprocessed files
    lc_files = glob.glob(os.path.join(processed_dir, "*_clean.npy"))
    
    data_list = []

    master_df = (
        catalog
        .merge(hon_data, on='KIC', how='left')
        .merge(general_catalog, on='KIC', how='left')  
    )

    for lc_path in lc_files:
        filename = os.path.basename(lc_path)
        try:
            # Extract KIC (digits 4 to 13)
            kic_id = int(filename[4:13])
            
            #Extract duration (looking for 20d, 55d, or 80d in filename)
            duration = "unknown"
            for d in ["20d", "55d", "80d"]:
                if d in filename:
                    duration = d
                    break

            match = master_df[master_df['KIC'] == kic_id]
            if not match.empty:
                evolstate=match.iloc[0]['EvolState']
                numax = match.iloc[0]['nu_max']
                numax_hon = match.iloc[0]['nu_max_hon']
                delta_nu= match.iloc[0]['delta_nu']
                teff = match.iloc[0]['Teff']
                psd_path = lc_path.replace("_clean.npy", "_psd.npy")
                numax_syd=match.iloc[0]['nu_max_syd']
                deltanu_syd=match.iloc[0]['delta_nu_syd']
                numax_a2z=match.iloc[0]['nu_max_a2z']
                deltanu_a2z=match.iloc[0]['delta_nu_a2z']
                numax_dia=match.iloc[0]['nu_max_dia']
                deltanu_dia=match.iloc[0]['delta_nu_dia']
                if os.path.exists(psd_path):
                    data_list.append({
                        'kic': kic_id,
                        'duration': duration,  
                        'lc_path': lc_path,
                        'psd_path': psd_path,
                        'evolstate': evolstate,
                        'nu_max': numax,
                        'nu_max_hon': numax_hon,
                        'delta_nu': delta_nu,
                        'nu_max_syd': numax_syd,        
                        'delta_nu_syd': deltanu_syd,
                        'nu_max_a2z': numax_a2z,
                        'delta_nu_a2z': deltanu_a2z,
                        'nu_max_dia': numax_dia,
                        'delta_nu_dia': deltanu_dia,
                        'teff': teff 
                    })
        except:
            continue

    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"Manifest created with {len(df)} stars.")

if __name__ == "__main__":
    create_manifest("./processed_ml_data_APOKASC3", "./table4APOKASC.txt", "./Table1Hon19.dat", "./pipelines_all.txt")
    df = pd.read_csv("./Baseline/manifest.csv")
    print(df[['nu_max', 'nu_max_hon', 'nu_max_syd']].describe())
    print(df[['delta_nu', 'delta_nu_syd']].describe())