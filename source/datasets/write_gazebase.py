import argparse
import csv
import itertools
import joblib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pymovements.paths import get_filepaths
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from checks import check_format_spec
from config import basepaths
from config import Config
from config.preprocessing import pipelines
from paths import ExperimentPaths, makedirs

from .preprocessor import preprocess_files
from .writer import write_data


experiment_sampling_rate = 1000

X_channels = {
    't': 0,
    'x': 1,
    'y': 2,
    'dx': 3,
    'dy': 4,
}

Y_columns = {
    'subject_id': 0,
    'round_id': 1,
    'session_id': 2,
    'task_id': 3,
}

data_key_format = 'gazebase_{selected_task}_sr{sampling_rate}_sl{sequence_length}_{pipeline_key}_dxy'

filename_regex = re.compile(
    r'S_(?P<round_id>\d)(?P<subject_id>\d+)'
    r'_S(?P<session_id>\d+)'
    r'_(?P<task_name>.+).csv'
)

task_name_map = {
    'BLG': 0,
    'FXS': 1,
    'HSS': 2,
    'RAN': 3,
    'TEX': 4,
    'VD1': 5,
    'VD2': 6,
}

selected_round_ids = [1, 2, 3, 4]
selected_session_ids = [1, 2]


def custom_preprocess_gazebase(
        df_file: pd.DataFrame,
) -> pd.DataFrame:
    # columns to be renamed in input csvs
    rename_csv_columns={"n": "t"}

    df_file.rename(columns=rename_csv_columns, inplace=True)
    return df_file


def get_subject_ids_with_first_n_rounds(n_rounds: int, df: pd.DataFrame):
    rounds_per_subject = {
        subject_id: set()
        for subject_id in df.subject_id.unique()
    }

    # first select subject ids
    groupby_iterator = df.groupby(['round_id', 'subject_id'])
    for (round_id, subject_id), group_df in groupby_iterator:
        rounds_per_subject[subject_id].add(round_id)

    subject_ids = []
    for subject_id, rounds_subject in rounds_per_subject.items():
        rounds_subject = sorted(rounds_subject)

        prev_round_id = 0
        for curr_round_id in rounds_subject[:n_rounds]:
            if curr_round_id != prev_round_id + 1:
                break
            elif curr_round_id == n_rounds:
                subject_ids.append(subject_id)
                break
            prev_round_id = curr_round_id

    return sorted(subject_ids)


def read_and_process_gazebase(
    pipeline_key: str,
    sequence_length: int,
    downsampling_factor: int,
    selected_label: str,
    selected_task: str,
    input_rootpath: Path,
):
    check_format_spec(X_channels)
    check_format_spec(Y_columns)

    print(f'{input_rootpath = }')
    print(f'{X_channels = }')
    print(f'{Y_columns = }')


    # map task name to task id
    if selected_task == 'all':
        selected_task_ids = [0, 1, 2, 3, 4, 5, 6]
    elif selected_task == 'allnoblg':
        # without BLG
        selected_task_ids = [1, 2, 3, 4, 5, 6]
    else:
        selected_task_ids = [task_name_map[selected_task.upper()]]
    print(f'{selected_task = }')
    print(f'{selected_task_ids = }')

    print(f'{selected_round_ids = }')
    print(f'{selected_session_ids = }')

    # get csv filepaths for all rounds
    round_dirpaths = [
        input_rootpath / f'Round_{round_id}'
        for round_id in selected_round_ids
    ]
    print(f'{round_dirpaths = }')

    csv_filepaths = []
    for round_dirpath in sorted(round_dirpaths):
        csv_filepaths += get_filepaths(round_dirpath, extension='.csv')

    print('number of csv files:', len(csv_filepaths))
    print(f'{csv_filepaths[0] = }')


    # parse experiment info from filename
    round_ids = [None for _ in csv_filepaths]
    subject_ids = [None for _ in csv_filepaths]
    session_ids = [None for _ in csv_filepaths]
    task_names = [None for _ in csv_filepaths]
    for idx, filepath in enumerate(csv_filepaths):

        match = filename_regex.match(filepath.name)

        if match:
            round_ids[idx] = match.group('round_id')
            subject_ids[idx] = match.group('subject_id')
            session_ids[idx] = match.group('session_id')
            task_names[idx] = match.group('task_name')

        else:
            raise Exception(f'filepath {filepath} has no parsing match')

    # create dataframe with all fileinfos
    fileinfo_df = pd.DataFrame({
        'round_id': round_ids,
        'subject_id': subject_ids,
        'session_id': session_ids,
        'task_name': task_names,
        'filepath': csv_filepaths,
    })

    fileinfo_df.subject_id = fileinfo_df.subject_id.astype(int)
    fileinfo_df.round_id = fileinfo_df.round_id.astype(int)
    fileinfo_df.session_id = fileinfo_df.session_id.astype(int)
    fileinfo_df['task_id'] = fileinfo_df.task_name.map(task_name_map)

    fileinfo_df.sort_values(by=['filepath'], inplace=True)
    print(fileinfo_df.task_name.unique())
    print(fileinfo_df.head())


    # only select subjects which were present in all first four rounds
    selected_subject_ids = get_subject_ids_with_first_n_rounds(
        n_rounds=4, df=fileinfo_df,
    )
    print(f'{selected_subject_ids = }')
    print('count:', len(selected_subject_ids))

    # select files for correct subjects , rounds and sessions
    fileinfo_df = fileinfo_df[
        (fileinfo_df.subject_id.isin(selected_subject_ids))
        & (fileinfo_df.round_id.isin(selected_round_ids))
        & (fileinfo_df.session_id.isin(selected_session_ids))
        & (fileinfo_df.task_id.isin(selected_task_ids))
    ]

    # get pipeline configuration
    pipeline_config = pipelines[pipeline_key]
    print(f'{pipeline_config = }')

    # preprocess files
    X, Y, events, event_properties = preprocess_files(
        fileinfo_df=fileinfo_df,
        X_channels=X_channels,
        Y_columns=Y_columns,
        input_sampling_rate=experiment_sampling_rate,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        pipeline_config=pipeline_config,
        custom_preprocess=custom_preprocess_gazebase,
    )

    print('unique subject ids:', np.unique(Y[:, Y_columns['subject_id']]))
    print('unique round ids:', np.unique(Y[:, Y_columns['round_id']]))
    print('unique session ids:', np.unique(Y[:, Y_columns['session_id']]))

    # convert class vector of numbers to binary class matrix
    label_encoder = LabelEncoder()
    label_encoder.fit(Y[:, Y_columns[selected_label]])
    labels = label_encoder.transform(Y[:, Y_columns[selected_label]])

    Y[:, Y_columns[selected_label]] = labels
    Y_subject_id = to_categorical(labels)
    print(f'{Y_subject_id.shape = }')


    # TODO: use selected_label variable with list to get Y_task instead of hardcoding
    # convert class vector of numbers to binary class matrix
    task_label_encoder = LabelEncoder()
    task_label_encoder.fit(Y[:, Y_columns['task_id']])
    task_labels = task_label_encoder.transform(Y[:, Y_columns['task_id']])

    Y[:, Y_columns['task_id']] = task_labels
    Y_task_id = to_categorical(task_labels)
    print(f'{Y_task_id.shape = }')

    # create folds with leave-one-round-out scheme:
    # validation set random sample of 10% of training data.
    val_frac = 0.1
    folds = {}
    fold_iterator = sorted(fileinfo_df.round_id.unique())
    for test_round_id in fold_iterator:
        fold_id = test_round_id

        mask_test = np.equal(Y[:, Y_columns['round_id']], test_round_id)

        # shuffle train/val set and use defined fraction for validation
        idx_train_val = np.where(~mask_test)[0]
        np.random.seed(test_round_id)
        np.random.shuffle(idx_train_val)
        n_val_instances = int(val_frac * len(idx_train_val))
        idx_val = idx_train_val[:n_val_instances]

        # create mask from idx
        mask_val = np.zeros(mask_test.shape, dtype=bool)
        mask_val[idx_val] = True

        mask_train = np.logical_not(np.logical_or(mask_test, mask_val))

        folds[fold_id] = {
            'test': mask_test,
            'train': mask_train,
            'val': mask_val,
        }

    return {
        'X': X,
        'Y_labels': Y,
        'Y_cat': Y_subject_id,
        'Y_task_id': Y_task_id,
        'X_format': X_channels,
        'Y_format': Y_columns,
        'events': events,
        'event_properties': event_properties,
        'folds': folds,
        'label_encoder': label_encoder,
        'task_id_encoder': task_label_encoder,
    }


def main(
    pipeline_key: str,
    sequence_length: int,
    downsampling_factor: int,
    selected_label: str,
    selected_task: str,
    input_rootpath: Path,
    output_rootpath: Path,
) -> None:
    if downsampling_factor != 1:
        raise NotImplementedError()
    output_sampling_rate = experiment_sampling_rate // downsampling_factor

    data = read_and_process_gazebase(
        pipeline_key=pipeline_key,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        selected_label=selected_label,
        selected_task=selected_task,
        input_rootpath=input_rootpath,
    )

    data_key = data_key_format.format(
        selected_task=selected_task,
        sampling_rate=output_sampling_rate,
        sequence_length=sequence_length,
        pipeline_key=pipeline_key,
    )
    print(f'{data_key = }')

    write_data(
        data=data,
        data_key=data_key,
        basepath=output_rootpath,
        additional_labels=['task_id'],
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequence_length', type=int, default=5000,
                        help='set sequence length')
    parser.add_argument('--downsampling', type=int, default=1,
                        help='set downsampling factor')
    parser.add_argument('--pipeline', type=str, default='savgol_maxvel1000',
                        help='set preprocessing pipeline')
    # available tasks: ran, hss, fxs, blg, tex, vd1, vd2, all, allnoblg
    parser.add_argument('--task', type=str, default='all',
                        help='set task name')
    parser.add_argument('--label', type=str, default='subject_id',
                        help='set label name')

    args = parser.parse_args()
    return {
        'sequence_length': args.sequence_length,
        'downsampling_factor': args.downsampling,
        'pipeline_key': args.pipeline,
        'selected_label': args.label,
        'selected_task': args.task,
    }


if __name__ == '__main__' :
    args = parse_args()
    main(
        input_rootpath=basepaths.gazebase,
        output_rootpath=basepaths.workspace,
         **args,
    )
