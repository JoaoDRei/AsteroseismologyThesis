import numpy as np
import torch
from torch.utils.data import Dataset

class AstroBaselineDataset(Dataset):
    def __init__(self, manifest_df, mode='lc', target_length=None):
        self.df = manifest_df
        self.mode = mode
        self.target_length = target_length or (65000 if mode == 'lc' else 35000)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = np.load(row['lc_path'] if self.mode == 'lc' else row['psd_path'])
        
        if self.mode == 'psd':
            freq = data[0]
            power = data[1]
            
            # 1. Physical Cutoff (Nyquist for Kepler is ~283)
            mask = (freq >= 0.1) & (freq <= 283)
            freq, power = freq[mask], power[mask]

            if len(freq) < 10:
                raise ValueError("Too few frequency points after masking")
            

            idx= np.argsort(freq)
            freq, power = freq[idx], power[idx]
            # 2. Log-Scale Power (helps the model see the 'bump') 
            power = np.log10(power + 1e-8)
            power = (power - np.mean(power)) / (np.std(power) + 1e-8) #standardization
            
            # 3. Physically meaningful interpolation
            freq_log=np.log10(freq+1e-8) #log-transform with small offset to avoid log(0)
            freq_log_new = np.linspace(freq_log.min(), freq_log.max(), self.target_length)
            y_final = np.interp(freq_log_new, freq_log, power)
            
        else: # Light Curve mode
            y = data[1]
            # 4. Random Cropping (better generalization)
            if len(y) > self.target_length:
                start = np.random.randint(0, len(y) - self.target_length)
                y_final = y[start : start + self.target_length]
            else:
                # Pad with 0 (which is the mean since pre-process normalization)
                y_final = np.pad(y, (0, self.target_length - len(y)), mode='constant')

        label = torch.tensor(row['nu_max']).float()
        return torch.tensor(y_final).float().unsqueeze(0), label, row['kic']

    def __len__(self):
        return len(self.df)