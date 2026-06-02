import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

class AstroBaselineDataset(Dataset):
    def __init__(self, manifest_df, mode='lc', target_length=None, deterministic=False):
        self.df = manifest_df
        self.mode = mode
        self.target_length = target_length or (65000 if mode == 'lc' else 35000)
        self.deterministic = deterministic
        with open("./Baseline/config.yaml", "r") as f:
                config = yaml.safe_load(f)

        self.config_targets = config['model']['targets']
        self.class_map = {"RGB": 0, "RC": 1}




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
            y_final=np.interp(freq_log_new, freq_log, power) #interpolate power at log-frequency points
            #power_interp = np.interp(freq_log_new, freq_log, power)

            #frequency channel
            #freq_interp=freq_log_new.copy()
            #freq_interp=(freq_interp-np.mean(freq_interp))/(np.std(freq_interp)+1e-8) #standardization
            #x=np.stack([power_interp,freq_interp],axis=0) #shape (2, target_length)
        else: # Light Curve mode
            y = data[1]
            # 4. Random Cropping (better generalization)
            if len(y) > self.target_length:
                if self.deterministic:
                    start = (len(y) - self.target_length) // 2  # always take the middle segment
                else:
                    start= np.random.randint(0, len(y) - self.target_length)
                y_final = y[start : start + self.target_length]
            else:
                # Pad with 0 (which is the mean since pre-process normalization)
                y_final = np.pad(y, (0, self.target_length - len(y)), mode='constant')
            #x=y_final[np.newaxis, :] #shape (1, target_length)
        #label = torch.tensor([row['nu_max'], row['delta_nu']]).float()
        targets = []

        for t in self.config_targets:
            if t == "nu_max":
                targets.append(np.log10(row['nu_max'] + 1e-8))

            elif t == "delta_nu":
                targets.append(np.log10(row['delta_nu'] + 1e-3))

            elif t == "class":
                ev = row["evolstate_norm"]
                targets.append(self.class_map[ev])

        x = torch.tensor(y_final, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(targets, dtype=torch.float32)

        return x, y, row['kic']
    def __len__(self):
        return len(self.df)