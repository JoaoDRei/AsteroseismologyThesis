import numpy as np
import os
from pathlib import Path

def check_x_axis_consistency(file_path):
    # Pathlib handles the / vs \ issue automatically
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: {path} not found.")
        return

    try:
        # We load with mmap_mode='r' to keep it fast
        data = np.load(path, mmap_mode='r')
        
        # Check if the data has the expected 2 rows (x and y)
        if data.ndim < 2 or data.shape[0] < 1:
            print(f"File {path.name} does not have an x-axis row at index 0.")
            return

        x_axis = data[0, :]  # Extract the first row
        total_len = len(x_axis)
        
        print(f"--- Consistency Check: {path.name} ---")
        print(f"Total Data Points: {total_len}")
        
        # Define indices to check
        indices = [0, 4999, 14999, total_len - 1] # 0-based indexing
        
        for idx in indices:
            if idx < total_len:
                val = x_axis[idx]
                # Using 1-based labels for the print to match your request
                label = "First" if idx == 0 else "Last" if idx == total_len-1 else f"{idx+1}th"
                print(f"  {label} element (index {idx}): {val}")
            else:
                print(f"  Index {idx} is out of bounds for this file.")
        
        print("-" * 45)

    except Exception as e:
        print(f"Error reading {path.name}: {e}")

if __name__ == "__main__":
    # Use Path to join folders and files—it solves the Linux/Windows slash issue!
    folder = "processed_ml_data_APOKASC3"
    filename = "kplr006303327_55d_clean.npy"
    target_path = Path(folder) / filename
    
    check_x_axis_consistency(target_path)
