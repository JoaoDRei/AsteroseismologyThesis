import numpy as np
import torch
from torch.utils.data import Dataset

class AstroBaselineDataset(Dataset):
    def __init__(self, manifest_df, mode='lc', target_length=None):
        self.df = manifest_df
        self.mode = mode
        # Set defaults if not provided
        if target_length is None:
            self.target_length = 65000 if mode == 'lc' else 35000
        else:
            self.target_length = target_length

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = row['lc_path'] if self.mode == 'lc' else row['psd_path']
        
        # Load [axis, values]
        data = np.load(file_path)
        y = data[1] # The flux or power
        
        if self.mode == 'psd':
            # INTERPOLATION: Stretch/shrink PSD to target_length
            x_old = np.linspace(0, 1, len(y))
            x_new = np.linspace(0, 1, self.target_length)
            y_final = np.interp(x_new, x_old, y)
        else:
            # PADDING/TRUNCATING: For Light Curves
            if len(y) > self.target_length:
                y_final = y[:self.target_length]
            else:
                y_final = np.pad(y, (0, self.target_length - len(y)), mode='constant')

        label = torch.tensor(row['nu_max']).float()
        return torch.tensor(y_final).float().unsqueeze(0), label # Shape: (1, Length)

    def __len__(self):
        return len(self.df)