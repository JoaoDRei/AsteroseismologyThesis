import pandas as pd
import os

# --- Settings ---
input_csv = "./CAIIHK_Activity_indexes_LAMOST_DR11_LRS.csv"
output_csv = "./kepler_activity_catalog.csv"

def extract_kepler_stars(file_path):
    print(f"Reading {file_path}...")
    
    # Using chunksize because the LAMOST DR11 catalog is likely very large (900k+ rows)
    # This prevents your computer from running out of memory
    chunk_list = []
    
    # Define the KIC ID range
    KIC_MIN = 750000
    KIC_MAX = 13000000

    # Iterate through the CSV in chunks
    for chunk in pd.read_csv(file_path, chunksize=50000):
        # 1. Ensure gp_id is numeric (ignore strings/errors)
        chunk['gp_id'] = pd.to_numeric(chunk['gp_id'], errors='coerce')
        
        # 2. Filter for IDs within the Kepler range
        # Also ensure we only keep rows where we actually have the activity labels (Rp_HK)
        kepler_mask = (chunk['gp_id'] >= KIC_MIN) & \
                      (chunk['gp_id'] <= KIC_MAX) & \
                      (chunk['Rp_HK_median'].notna())
        
        kepler_chunk = chunk[kepler_mask].copy()
        
        if not kepler_chunk.empty:
            chunk_list.append(kepler_chunk)
            print(f"Found {len(kepler_chunk)} Kepler stars in this chunk...")

    if not chunk_list:
        print("No Kepler stars found. Check if gp_id contains KIC IDs or Gaia IDs.")
        return

    # Combine all found chunks
    full_kepler_df = pd.concat(chunk_list)
    
    # Clean up: Convert KIC to integer and rename for your pipeline
    full_kepler_df['gp_id'] = full_kepler_df['gp_id'].astype(int)
    full_kepler_df.rename(columns={'gp_id': 'KIC'}, inplace=True)
    
    # Save the new CSV
    full_kepler_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved {len(full_kepler_df)} stars to {output_csv}")
    print("These are ready to be matched with your KEPSEISMIC files.")

if __name__ == "__main__":
    if os.path.exists(input_csv):
        extract_kepler_stars(input_csv)
    else:
        print(f"Error: {input_csv} not found in the current directory.")