import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticPSDPeakDataset(Dataset):
    def __init__(self, n_samples=1000, length=None):
        if length is None:
            length = 1000  # safe default for sanity tests

        self.n_samples = n_samples
        self.length = int(length)
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        x = np.linspace(0, 500, self.length)

        peak_loc = np.random.uniform(50, 450)

        # noise + gaussian peak
        y = np.random.normal(0, 0.1, self.length)
        y += np.exp(-(x - peak_loc)**2 / (2 * 15**2))

        # normalize (like your PSD pipeline)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

        x_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor([peak_loc], dtype=torch.float32)

        return x_t, y_t, idx