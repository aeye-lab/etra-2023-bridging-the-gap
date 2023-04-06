from collections.abc import Sequence
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pymovements.transforms import cut_into_subsequences
from pymovements.transforms import downsample
from pymovements.transforms import pos2vel
from pymovements.transforms import vnorm

from datasets.loader import DataFold


def get_monocular_statistics(x, statistic='mean'):
    """
    Compute zscore-statistics while preserving binocular differences.
    Doesn't raise warnings on monocular data or single channel time series.
    """
    if statistic == 'mean':
        statistic = np.nanmean
    elif statistic == 'std':
        statistic = np.nanstd

    value = statistic(x, axis=(0, 2), keepdims=True)
    if value.shape[1] == 4:
        value[:, 2:, :] = value[:, :2, :]
    return value


def del_fast_trafo(
    x: np.ndarray,
    v_min: float,
    v_replace: Union[float, Sequence],
) -> np.ndarray:
    # we expect batched data in channel first format
    n_channels = x.shape[1]
    sequence_length = x.shape[2]

    # check if v_replace is a scalar create array of n_channel elements.
    if isinstance(v_replace, (int, float)):
        v_replace = np.repeat(v_replace, n_channels)

    if n_channels in [1, 2]:
        pass  # we will do  the transform in the next block
    elif n_channels == 4:
        # apply recursively on each eye
        x_right = x[:, :2, :]
        x_left = x[:, 2:, :]

        x_right = del_fast_trafo(x_right, v_min=v_min, v_replace=v_replace[:2])
        x_left = del_fast_trafo(x_left, v_min=v_min, v_replace=v_replace[2:])

        # concatenate accordingly and return
        x_trafo = np.concatenate([x_right, x_left], axis=1)
        return x_trafo

    else:
        raise ValueError("only one, two or four channels supported")

    # create mask for replacing with v_replace
    v_abs = vnorm(x)
    mask = v_abs < v_min
    mask = np.tile(mask[:, :, np.newaxis], 2)
    mask = np.swapaxes(mask, 1, 2)

    # bring v_replace into correct shape (n_channels, sequence_length)
    v_replace = np.repeat(v_replace[:, np.newaxis], sequence_length, axis=1)

    # apply mask to x and replace with v_replace
    x_trafo = np.where(mask, v_replace, x)
    return x_trafo


def del_slow_trafo(x: np.ndarray, c: float) -> np.ndarray:
    x_trafo = np.tanh(x * c)
    return x_trafo


def preprocess_datafold(
        fold: DataFold,
        method: str,
        flatten_channels: bool = False,
        **kwargs,
) -> DataFold:
    '''
    This is used for input transformations.
    '''

    print("preprocessing fold.")

    # default pass needed for simple error-catch in else-clause
    if method is None:
        pass

    elif method == 'del-merged':
        # we will concatenate fast and slow subnet input on the channel axis
        # this results in a tensor with twice as many channels as originally.
        # the fast/slow subnet of the deepeyedeintification live model
        # will then use the respective channels as input.

        # clip transform for fast subnet input
        v_min = kwargs['v_min']
        X_test_fast = del_fast_trafo(fold.X_test, v_min=v_min, v_replace=0)
        X_train_fast = del_fast_trafo(fold.X_train, v_min=v_min, v_replace=0)
        X_val_fast = del_fast_trafo(fold.X_val, v_min=v_min, v_replace=0)

        # z-score normalization from stats of first two channels only.
        # this way eye differences are retained.
        mean = get_monocular_statistics(X_train_fast, 'mean')
        std = get_monocular_statistics(X_train_fast, 'std')

        X_test_fast = (X_test_fast - mean) / std
        X_train_fast = (X_train_fast - mean) / std
        X_val_fast = (X_val_fast - mean) / std

        # preprocess slow subnet input
        c = kwargs['c']
        X_test_slow = del_slow_trafo(fold.X_test, c=c)
        X_train_slow = del_slow_trafo(fold.X_train, c=c)
        X_val_slow = del_slow_trafo(fold.X_val, c=c)

        # concatenate channels for each subnet input
        fold.X_test = np.concatenate([X_test_fast, X_test_slow], axis=1)
        fold.X_train = np.concatenate([X_train_fast, X_train_slow], axis=1)
        fold.X_val = np.concatenate([X_val_fast, X_val_slow], axis=1)

    elif method == 'del-merged-zstd':
        # z-score normalization from stats of first two channels only.
        # this way eye differences are retained.
        mean = get_monocular_statistics(fold.X_train, 'mean')
        std = get_monocular_statistics(fold.X_train, 'std')

        fold.X_test = (fold.X_test - mean) / std
        fold.X_train = (fold.X_train - mean) / std
        fold.X_val = (fold.X_val - mean) / std

        # tile input channels for each subnet
        fold.X_test = np.tile(fold.X_test, (1, 2, 1))
        fold.X_train = np.tile(fold.X_train, (1, 2, 1))
        fold.X_val = np.tile(fold.X_val, (1, 2, 1))

    elif method == 'del-merged-zstd-tanh':
        # z-score normalization from stats of first two channels only.
        # this way eye differences are retained.
        mean = get_monocular_statistics(fold.X_train, 'mean')
        std = get_monocular_statistics(fold.X_train, 'std')

        X_test_fast = (fold.X_test - mean) / std
        X_train_fast = (fold.X_train - mean) / std
        X_val_fast = (fold.X_val - mean) / std

        # preprocess slow subnet input
        c = kwargs['c']
        X_test_slow = del_slow_trafo(fold.X_test, c=c)
        X_train_slow = del_slow_trafo(fold.X_train, c=c)
        X_val_slow = del_slow_trafo(fold.X_val, c=c)

        # concatenate channels for each subnet input
        fold.X_test = np.concatenate([X_test_fast, X_test_slow], axis=1)
        fold.X_train = np.concatenate([X_train_fast, X_train_slow], axis=1)
        fold.X_val = np.concatenate([X_val_fast, X_val_slow], axis=1)

    elif method == 'del-fast-trafo':
        fold.X_test = del_fast_trafo(fold.X_test, v_replace=0, **kwargs)
        fold.X_train = del_fast_trafo(fold.X_train, v_replace=0, **kwargs)
        fold.X_val = del_fast_trafo(fold.X_val, v_replace=0, **kwargs)

        mean = get_monocular_statistics(fold.X_train, 'mean')
        std = get_monocular_statistics(fold.X_train, 'std')

        fold.X_test = (fold.X_test - mean) / std
        fold.X_train = (fold.X_train - mean) / std
        fold.X_val = (fold.X_val - mean) / std

    elif method == 'del-slow-trafo':
        fold.X_test = del_slow_trafo(fold.X_test, **kwargs)
        fold.X_train = del_slow_trafo(fold.X_train, **kwargs)
        fold.X_val = del_slow_trafo(fold.X_val, **kwargs)

    elif method.startswith('zstd'):
        if method == 'zstd':
            # z-score normalization with train-set statistics
            mean = fold.X_train.nanmean(axis=(0, 2), keepdims=True)
            std = fold.X_train.nanstd(axis=(0, 2), keepdims=True)
        elif method == 'zstd-mono':
            # z-score normalization on first two channels only
            mean = get_monocular_statistics(fold.X_train, 'mean')
            std = get_monocular_statistics(fold.X_train, 'std')
        else:
            raise ValueError('method setting "{method}" not valid')

        fold.X_test = (fold.X_test - mean) / std
        fold.X_train = (fold.X_train - mean) / std
        fold.X_val = (fold.X_val - mean) / std
    else:
        raise ValueError('method setting "{method}" not valid')

    # replace all nans with zero
    nan_to_num_kwargs = {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0}
    fold.X_test = np.nan_to_num(fold.X_test, **nan_to_num_kwargs)
    fold.X_train = np.nan_to_num(fold.X_train, **nan_to_num_kwargs)
    fold.X_val = np.nan_to_num(fold.X_val, **nan_to_num_kwargs)

    if flatten_channels:
        flattened_size = fold.X_test[0].size
        fold.X_test = fold.X_test.reshape(len(fold.X_test), 1, flattened_size)
        fold.X_train = fold.X_train.reshape(len(fold.X_train), 1, flattened_size)
        fold.X_val = fold.X_val.reshape(len(fold.X_val), 1, flattened_size)

    # we changed the underlying data and need to initialize all dataloaders again
    # TODO: do this automatically in DataFold class on setting X/Y attributes.
    fold.init_dataloaders()

    return fold
