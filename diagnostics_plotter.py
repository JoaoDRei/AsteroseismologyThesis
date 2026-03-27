import lightkurve as lk
import matplotlib.pyplot as plt
import glob
import os

def plot_with_lightkurve(kic_id, base_dir="./kepseismic_data", filter_type="80d"):
    # 1. Find the file
    kic_padded = str(int(kic_id)).zfill(9)
    # The glob pattern now correctly uses the filter_type variable
    search_pattern = os.path.join(base_dir, "**", f"*{kic_padded}*{filter_type}*filt-inp.fits")
    
    print(f"Searching for: {search_pattern}")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"!!! Could not find KIC {kic_id} with filter {filter_type} in {base_dir}")
        return

    # 2. Load the Light Curve
    lc = lk.read(files[0])
    
    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Time Domain
    lc.plot(ax=ax1, color='black', lw=0.5)
    ax1.set_title(f"KIC {kic_id} - Time Domain ({filter_type} Filter)")
    
    # 4. Create Periodogram 
    pg = lc.to_periodogram(method='lombscargle', normalization='psd')
    
    # 5. Plot Periodogram
    pg.plot(ax=ax2, color='royalblue', lw=0.8)
    
    # Standard Asteroseismic View: Log-Log scale
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(10, 5000) 
    ax2.set_title(f"Frequency Domain: PSD ({filter_type})")
    
    plt.tight_layout()
    plt.show()

# --- CORRECT WAYS TO CALL ---

# Option A: Use the default base_dir but change the filter
plot_with_lightkurve(10010623, filter_type="55d")

