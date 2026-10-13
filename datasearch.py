import lightkurve as lk
import matplotlib.pyplot as plt

# KIC 11446443 is a great overlap star (Red Giant)
target = "KIC 11446443" 

print(f"Searching for {target}...")

# 1. Search Kepler - Get the first available quarter
search_kepler = lk.search_lightcurve(target, mission="Kepler", author="Kepler")
if len(search_kepler) > 0:
    lc_kepler = search_kepler[0].download()
else:
    lc_kepler = None
    print("No Kepler data found.")

# 2. Search TESS - Get the first available sector
search_tess = lk.search_lightcurve(target, mission="TESS", author="SPOC")
if len(search_tess) > 0:
    lc_tess = search_tess[0].download()
else:
    lc_tess = None
    print("No TESS data found.")

# 3. Plotting (Only if both were found)
if lc_kepler and lc_tess:
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # We use .remove_nans() to avoid plotting errors
    lc_kepler.remove_nans().normalize().scatter(ax=ax[0], color='blue', label="Kepler")
    lc_tess.remove_nans().normalize().scatter(ax=ax[1], color='red', label="TESS")
    
    ax[0].set_title(f"Kepler Data (Clean)")
    ax[1].set_title(f"TESS Data (Noisier)")
    plt.tight_layout()
    plt.show()
    print("Successfully plotted")
else:
    print("Could not find overlapping data for this target.")