import numpy as np
import torch
from torch.utils.data import Dataset

from config import AugmentationConfig


class AugmentedDataset(Dataset):
    def __init__(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            config: AugmentationConfig,
    ):
        assert len(X) == len(Y)

        self.X = X
        self.Y = Y

        self.n_original = len(X)  # number of original instances
        self.n_augmentations = config.n_augmentations
        self.noise_std = config.noise_std

    def __len__(self):
        return self.n_original * (1 + self.n_augmentations)

    def __getitem__(self, idx):
        if not np.issubdtype(type(idx), np.integer):
            raise ValueError(f"{idx = }, type: {type(idx)}")
        if idx >= self.__len__():
            raise ValueError()

        # get original instance
        original_idx = idx % self.n_original
        x = self.X[original_idx].clone()
        y = self.Y[original_idx].clone()

        # create additive gaussian noise if not in the first N
        if idx >= self.n_original:
            x += torch.randn_like(x) * self.noise_std

        return x, y


        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # if isinstance(idx, int):
        #     idx = [idx]

        # # get idx of original instance without noise
        # original_idx = [idx_i % len(self.X) for idx_i in idx]

        # x_idx = self.X[original_idx]
        # y_idx = self.Y[original_idx]

        # for i, idx_i in enumerate(idx):
        #     # this means idx_i is an instance without noise, just continue
        #     if idx_i < len(self.X):
        #         continue

        #     # else, create additive gaussian noise with noise_std
        #     x_idx[i] += torch.randn_like(x_idx[i]) * self.noise_std

        # return x_idx, y_idx
