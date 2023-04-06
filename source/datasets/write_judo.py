import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pymovements.base import Experiment
from pymovements.paths import get_filepaths
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from checks import check_format_spec
from config import basepaths
from config.preprocessing import pipelines
from .preprocessor import preprocess_files
from .writer import write_data


experiment = Experiment(
    screen_width_px=1280,
    screen_height_px=1024,
    screen_width_cm=38,
    screen_height_cm=30.2,
    distance_cm=68,
    sampling_rate=1000,
)

X_channels = {
    't': 0,
    'xr': 1,
    'yr': 2,
    'xl': 3,
    'yl': 4,
    'dxr': 5,
    'dyr': 6,
    'dxl': 7,
    'dyl': 8,
}

Y_columns = {
    'subject_id': 0,
    'session_id': 1,
    'trial_id': 2,
}

data_key_format = 'judo_sr{sampling_rate}_sl{sequence_length}_{pipeline_key}_bxy'

filename_regex = re.compile(
    r'(?P<subject_id>\d+)'
    r'_(?P<session_id>\d+).csv'
)


def custom_preprocess_judo(
    df_file: pd.DataFrame,
) -> pd.DataFrame:
    df_file.rename(
        columns={
            'time': 't',
            'trialId': 'trial_id',
            'x_right': 'xr',
            'y_right': 'yr',
            'x_left': 'xl',
            'y_left': 'yl',
        },
        inplace=True,
    )
    df_file.t = df_file.t.astype(float)
    df_file.xr = df_file.xr.astype(float)
    df_file.yr = df_file.yr.astype(float)
    df_file.xl = df_file.xl.astype(float)
    df_file.yl = df_file.yl.astype(float)

    pos_cols = ['xr', 'yr', 'xl', 'yl']
    df_file[pos_cols] = experiment.screen.pix2deg(
        df_file[pos_cols], center_origin=True)

    return df_file


def read_and_process_judo(
    pipeline_key: str,
    sequence_length: int,
    downsampling_factor: int,
    selected_label: str,
    input_rootpath: Path,
):
    check_format_spec(X_channels)
    check_format_spec(Y_columns)

    print(f'{input_rootpath = }')
    print(f'{X_channels = }')
    print(f'{Y_columns = }')

    # parse experiment info from filename
    csv_filepaths = get_filepaths(
        rootpath=input_rootpath,
        regex=filename_regex,
    )
    print('number of csv files:', len(csv_filepaths))
    print(f'{csv_filepaths[0] = }')


    subject_ids = [None for _ in csv_filepaths]
    session_ids = [None for _ in csv_filepaths]
    for idx, filepath in enumerate(csv_filepaths):

        match = filename_regex.match(filepath.name)

        if match:
            subject_ids[idx] = match.group('subject_id')
            session_ids[idx] = match.group('session_id')

        else:
            raise Exception(f'filepath {filepath} has no parsing match')

    # create dataframe with all fileinfos
    fileinfo_df = pd.DataFrame({
        'subject_id': subject_ids,
        'session_id': session_ids,
        'filepath': csv_filepaths,
    })

    fileinfo_df.subject_id = fileinfo_df.subject_id.astype(int)
    fileinfo_df.session_id = fileinfo_df.session_id.astype(int)
    fileinfo_df.sort_values(by=['filepath'], inplace=True)

    print("unique subject ids:", fileinfo_df.subject_id.unique())
    print("unique session ids:", fileinfo_df.session_id.unique())
    print(fileinfo_df.head())

    # get pipeline configuration
    pipeline_config = pipelines[pipeline_key]
    print(f'{pipeline_config = }')

    # preprocess files
    X, Y, events, event_properties = preprocess_files(
        fileinfo_df=fileinfo_df,
        X_channels=X_channels,
        Y_columns=Y_columns,
        csv_groupby='trial_id',
        input_sampling_rate=experiment.sampling_rate,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        pipeline_config=pipeline_config,
        read_csv_kwargs={'delim_whitespace': True, 'na_values': ''},
        custom_preprocess=custom_preprocess_judo,
    )

    print('unique subject ids:', np.unique(Y[:, Y_columns['subject_id']]))
    print('unique session ids:', np.unique(Y[:, Y_columns['session_id']]))
    print('unique trial ids:', np.unique(Y[:, Y_columns['trial_id']]))

    # convert class vector of numbers to binary class matrix
    label_encoder = LabelEncoder()
    label_encoder.fit(Y[:, Y_columns[selected_label]])
    labels = label_encoder.transform(Y[:, Y_columns[selected_label]])

    Y[:, Y_columns[selected_label]] = labels
    Y_cat = to_categorical(labels)
    print(f'{Y_cat.shape = }')

    # validation set are last 12 trials from each training session
    trial_ids_train = np.arange(1, 144-12+1)
    trial_ids_val = np.arange(144-12, 144+1)

    # create folds with leave-one-text-out scheme:
    folds = {}
    fold_iterator = sorted(fileinfo_df.session_id.unique())
    for test_session_id in fold_iterator:
        fold_id = test_session_id

        mask_test = np.equal(Y[:, Y_columns['session_id']], test_session_id)

        mask_train = np.logical_and(
            np.logical_not(mask_test),
            np.isin(Y[:, Y_columns['trial_id']], trial_ids_train),
        )
        mask_val = np.logical_and(
            np.logical_not(mask_test),
            np.isin(Y[:, Y_columns['trial_id']], trial_ids_val),
        )

        folds[fold_id] = {
            'test': mask_test,
            'train': mask_train,
            'val': mask_val,
        }

    return {
        'X': X,
        'Y_labels': Y,
        'Y_cat': Y_cat,
        'X_format': X_channels,
        'Y_format': Y_columns,
        'events': events,
        'event_properties': event_properties,
        'folds': folds,
        'label_encoder': label_encoder,
    }


def main(
    pipeline_key: str,
    sequence_length: int,
    downsampling_factor: int,
    selected_label: str,
    input_rootpath: Path,
    output_rootpath: Path,
) -> None:
    data = read_and_process_judo(
        pipeline_key=pipeline_key,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        selected_label=selected_label,
        input_rootpath=input_rootpath,
    )

    output_sampling_rate = experiment.sampling_rate // downsampling_factor

    data_key = data_key_format.format(
        sampling_rate=output_sampling_rate,
        sequence_length=sequence_length,
        pipeline_key=pipeline_key,
    )
    print(f'{data_key = }')

    write_data(
        data=data,
        data_key=data_key,
        basepath=output_rootpath,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequence_length', type=int, default=1000,
                        help='set sequence length')
    parser.add_argument('--downsampling', type=int, default=1,
                        help='set downsampling factor')
    parser.add_argument('--pipeline', type=str, default='savgol_maxvel1000',
                        help='set preprocessing pipeline')
    parser.add_argument('--label', type=str, default='subject_id',
                        help='set label name')

    args = parser.parse_args()
    return {
        'sequence_length': args.sequence_length,
        'downsampling_factor': args.downsampling,
        'pipeline_key': args.pipeline,
        'selected_label': args.label,
    }


if __name__ == '__main__' :
    args = parse_args()
    main(
        input_rootpath=basepaths.judo,
        output_rootpath=basepaths.workspace,
         **args,
    )
