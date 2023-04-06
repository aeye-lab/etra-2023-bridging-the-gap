import joblib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import Sampler

from config import AugmentationConfig
from datasets.augmentation import AugmentedDataset


def load_xy_data(X_filepath: str, Y_filepath: str, X_format_filepath: str,
                 selected_input_channels: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, Dict]:

    X_full = np.load(X_filepath)
    Y = np.load(Y_filepath)
    with open(X_format_filepath, 'r') as f:
        X_channels_full = json.load(f)

    selected_channel_idxs = [
        X_channels_full[channel] for channel in selected_input_channels
    ]
    X_channels = {
        channel: new_channel_id
        for new_channel_id, channel in enumerate(selected_input_channels)
    }

    X = X_full[:, :, selected_channel_idxs]

    return X, Y, X_channels, X_full, X_channels_full


def load_label_data(Y_labels_filepath: str, Y_format_filepath: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    Y_labels = np.load(Y_labels_filepath)
    with open(Y_format_filepath, 'r') as f:
        Y_columns = json.load(f)

    return Y_labels, Y_columns


def load_event_data(event_filepaths: List[str]) -> pd.DataFrame:
    event_dfs = []
    for event_filepath in event_filepaths:
        event_df = pd.read_csv(event_filepath)
        event_filename = os.path.basename(event_filepath)
        event_type = os.path.splitext(event_filename)[0]
        event_df['event_type'] = event_type
        event_df.instance_id = event_df.instance_id.astype(int)

        event_dfs.append(event_df)

    return pd.concat(event_dfs)


def load_data(config, paths):
    # load data
    selected_input_channels = config.dataset['X_channels']
    X, Y, X_channels, X_full, X_channels_full = load_xy_data(
        X_filepath=paths['X'],
        Y_filepath=paths['Y'],
        X_format_filepath=paths['X_format'],
        selected_input_channels=selected_input_channels,
    )

    Y_labels, Y_columns = load_label_data(
        Y_labels_filepath=paths['Y_labels'],
        Y_format_filepath=paths['Y_format'],
    )

    # load fold indices
    folds = joblib.load(paths['folds'])

    print("Selected input channels:", selected_input_channels)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Folds:", list(folds.keys()))

    # swap axes for pytorch
    # tensorflow: channel axis is last axis (datasets are saved in this layout)
    # pytorch: channel axis is first axis (after batch axis)
    X = np.swapaxes(X, 1, 2)

    return {
        'X': X,
        'Y': Y,
        'Y_labels': Y_labels,
        'X_channels': X_channels,
        'Y_columns': Y_columns,
        'X_full': X_full,
        'X_channels_full': X_channels_full,
        'folds': folds,
    }


class DataFold:

    def __init__(
            self,
            fold_id,
            X_train, Y_train, idx_train,
            X_test, Y_test, idx_test,
            X_val, Y_val, idx_val,
            batch_size,
            augmentation_config,
            n_workers,
            training_sampler: Sampler = None,
    ):
        self.id = fold_id

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val

        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_val = Y_val

        self.idx_train = idx_train
        self.idx_test = idx_test
        self.idx_val = idx_val

        self.batch_size = batch_size
        self.augmentation_config = augmentation_config
        self.training_sampler = training_sampler

        self.n_workers = n_workers

        self.init_dataloaders()

    def init_dataloaders(self):
        # create dataloaders for train, validation and test data
        self.train_ds = AugmentedDataset(
            X=torch.from_numpy(self.X_train).type(torch.FloatTensor),
            Y=torch.from_numpy(self.Y_train).type(torch.FloatTensor),
            config=self.augmentation_config,
        )

        self.init_train_dataloader()

        self.val_ds = TensorDataset(
            torch.from_numpy(self.X_val).type(torch.FloatTensor),
            torch.from_numpy(self.Y_val).type(torch.FloatTensor))
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size,
                                 num_workers=self.n_workers)

        self.test_ds = TensorDataset(
            torch.from_numpy(self.X_test).type(torch.FloatTensor),
            torch.from_numpy(self.Y_test).type(torch.FloatTensor))
        self.test_dl = DataLoader(self.test_ds, batch_size=self.batch_size,
                                  num_workers=self.n_workers)

    def init_train_dataloader(self):
        # batch normalization layers can create errors when trained
        # with a batch size of one. if the last batch will have just a
        # single instance, we will simply drop this single instance
        # from training. each training epoch will be reshuffled, so it
        # will be probably a different instance in each epoch.
        drop_last = False
        n_instances = len(self.train_ds)
        if n_instances % self.batch_size == 1:
            drop_last = True

        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.training_sampler is None else False,
            num_workers=self.n_workers,
            sampler=self.training_sampler,
            drop_last=drop_last,
        )

    def set_training_sampler(self, sampler: Sampler):
        self.training_sampler = sampler
        self.init_train_dataloader()

    def get_labels(self, set_name: str) -> torch.Tensor:
        if set_name == 'test':
            dataset = self.train_ds
        elif set_name == 'val':
            dataset = self.val_ds
        elif set_name == 'train':
            dataset = self.train_ds

        catmat = torch.stack(
            [dataset[i][1] for i in range(len(dataset))], dim=0,
        )
        labels = torch.argmax(catmat, dim=-1)
        return labels


class DataFoldFactory:

    def __init__(
            self,
            X, Y, Y_labels, folds,
            X_channels, Y_columns,
            X_full, X_channels_full,
            batch_size: int = 64,
            augmentation_config: Optional[AugmentationConfig] = None,
            n_workers: int = 4,
    ):
        self.X = X
        self.Y = Y

        self.Y_labels = Y_labels
        self.X_channels = X_channels
        self.Y_columns = Y_columns

        self.X_full = X_full
        self.X_channels_full = X_channels_full

        self.folds = folds
        self.next_folds = list(folds.keys())

        self.batch_size = batch_size
        self.augmentation_config = augmentation_config
        self.n_workers = n_workers

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.folds)

    def __next__(self):
        # first iteration, create background
        if not self.next_folds:
            raise StopIteration

        fold_id = self.next_folds.pop(0)
        idxs = self.folds[fold_id]
        idx_test, idx_train, idx_val = idxs['test'], idxs['train'], idxs['val']
        X_test, X_train, X_val = self.X[idx_test], self.X[idx_train], self.X[idx_val]
        Y_test, Y_train, Y_val = self.Y[idx_test], self.Y[idx_train], self.Y[idx_val]

        datafold = DataFold(
            fold_id=fold_id,
            X_test=X_test, Y_test=Y_test, idx_test=idx_test,
            X_train=X_train, Y_train=Y_train, idx_train=idx_train,
            X_val=X_val, Y_val=Y_val, idx_val=idx_val,
            batch_size=self.batch_size,
            augmentation_config=self.augmentation_config,
            n_workers=self.n_workers,
        )

        return datafold
