import pandas as pd
import glob
import os
import numpy as np

def create_manifest(processed_dir, catalog_path,hon_path, general_catalog_path, output_csv="./Baseline/manifest.csv"):
    # 1. Load the APOKASC catalog
    # Adjust names/colspecs based on your specific table4APOKASC.txt format
    names = [
        'KIC', 'EvolState', 'quality_classification', 'nu_max', 'delta_nu',
        'delta_nu_correction_factor', 'mass', 'mass_err', 'radius',
        'radius_err', 'logg', 'logg_err', 'logg_spec', 'logg_spec_err',
        'Teff', 'metallicity'
    ]
    colspecs = [
        (0, 8), (9, 16), (22, 30), (49, 59), (71, 81), (93, 103),
        (126, 136), (137, 147), (148, 158), (159, 169), (170, 180),
        (181, 191), (214, 224), (225, 235), (192, 202), (236, 246)
    ]
    catalog = pd.read_fwf(catalog_path, colspecs=colspecs, names=names, skiprows=111)
    catalog.replace(-9999, np.nan, inplace=True)
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
            metadata = match.iloc[0] if not match.empty else {}
            psd_path = lc_path.replace("_clean.npy", "_psd.npy")
            if os.path.exists(psd_path):
                data_list.append({
                    'kic': kic_id,
                    'duration': duration,
                    'lc_path': lc_path,
                    'psd_path': psd_path,
                    'evolstate': metadata.get('EvolState', np.nan),
                    'nu_max': metadata.get('nu_max', np.nan),
                    'nu_max_hon': metadata.get('nu_max_hon', np.nan),
                    'delta_nu': metadata.get('delta_nu', np.nan),
                    'nu_max_syd': metadata.get('nu_max_syd', np.nan),
                    'delta_nu_syd': metadata.get('delta_nu_syd', np.nan),
                    'nu_max_a2z': metadata.get('nu_max_a2z', np.nan),
                    'delta_nu_a2z': metadata.get('delta_nu_a2z', np.nan),
                    'nu_max_dia': metadata.get('nu_max_dia', np.nan),
                    'delta_nu_dia': metadata.get('delta_nu_dia', np.nan),
                    'logg': metadata.get('logg', np.nan),
                    'logg_seis': metadata.get('logg', np.nan),
                    'logg_err': metadata.get('logg_err', np.nan),
                    'logg_spec': metadata.get('logg_spec', np.nan),
                    'logg_spec_err': metadata.get('logg_spec_err', np.nan),
                    'mass': metadata.get('mass', np.nan),
                    'mass_err': metadata.get('mass_err', np.nan),
                    'radius': metadata.get('radius', np.nan),
                    'radius_err': metadata.get('radius_err', np.nan),
                    'quality_classification': metadata.get('quality_classification', np.nan),
                    'delta_nu_correction_factor': metadata.get('delta_nu_correction_factor', np.nan),
                    'teff': metadata.get('Teff', np.nan),
                    'metallicity': metadata.get('metallicity', np.nan)
                })
        except:
            continue

    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"Manifest created with {len(df)} stars.")

if __name__ == "__main__":
    create_manifest("./processed_ml_data_APOKASC3", "./Baseline/Databases/table4APOKASC.txt", "./Baseline/Databases/Table1Hon19.dat", "./Baseline/Databases/pipelines_all.txt")
    df = pd.read_csv("./Baseline/manifest.csv")
    print(df[['nu_max', 'nu_max_hon', 'nu_max_syd']].describe())
    print(df[['delta_nu', 'delta_nu_syd']].describe())