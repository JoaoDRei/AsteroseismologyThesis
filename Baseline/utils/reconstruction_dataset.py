import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AstroReconstructionDataset(Dataset):
    """
    Dataset for CNN-BiLSTM reconstruction experiments.

    Uses the same light-curve preprocessing as AstroBaselineDataset:

        - load LC
        - random crop if longer than target_length
        - deterministic middle crop for validation/test
        - zero-pad if shorter

    The reconstruction target is a coarse-grained version of
    the processed light curve obtained with adaptive average pooling.

    Returns:

        x:
            Input light curve
            Shape: (1, target_length)

        target:
            Coarse reconstruction target
            Shape: (reconstruction_length,)

        kic:
            KIC identifier
    """

    def __init__(
        self,
        manifest_df,
        target_length=65000,
        reconstruction_length=4096,
        deterministic=False
    ):
        self.df = manifest_df.reset_index(drop=True)

        self.target_length = target_length
        self.reconstruction_length = reconstruction_length

        self.deterministic = deterministic

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ---------------------------------------------------------
        # Load light curve
        # ---------------------------------------------------------

        data = np.load(
            row["lc_path"]
        )

        y = data[1]

        # ---------------------------------------------------------
        # Crop / pad
        # ---------------------------------------------------------

        if len(y) > self.target_length:

            if self.deterministic:

                # Same deterministic behaviour as
                # AstroBaselineDataset

                start = (
                    len(y) - self.target_length
                ) // 2

            else:

                # Same random-crop behaviour as
                # AstroBaselineDataset

                start = np.random.randint(
                    0,
                    len(y) - self.target_length
                )

            y_final = y[
                start:start + self.target_length
            ]

        else:

            # Same zero-padding behaviour as
            # AstroBaselineDataset

            y_final = np.pad(
                y,
                (
                    0,
                    self.target_length - len(y)
                ),
                mode="constant"
            )

        # ---------------------------------------------------------
        # Convert to tensor
        # ---------------------------------------------------------

        x = torch.tensor(
            y_final,
            dtype=torch.float32
        ).unsqueeze(0)

        # Shape:
        #
        # (1, 65000)

        # ---------------------------------------------------------
        # Create reconstruction target
        # ---------------------------------------------------------

        target = F.adaptive_avg_pool1d(
            x.unsqueeze(0),
            self.reconstruction_length
        )

        # Before squeeze:
        #
        # (1, 1, 4096)

        target = target.squeeze(0).squeeze(0)

        # Shape:
        #
        # (4096,)

        return x, target, row["kic"]

    def __len__(self):
        return len(self.df)