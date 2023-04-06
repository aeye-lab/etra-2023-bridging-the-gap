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
    screen_width_px=1680,
    screen_height_px=1050,
    screen_width_cm=47.5,
    screen_height_cm=30,
    distance_cm=61,
    sampling_rate=1000,
)

X_channels = {
    't': 0,
    'x': 1,
    'y': 2,
    'dx': 3,
    'dy': 4,
}

Y_columns = {
    'subject_id': 0,
    'text_id': 1,
    'trial_id': 2,
}

data_key_format = 'potec_sr{sampling_rate}_sl{sequence_length}_{pipeline_key}_dxy'

filename_regex = re.compile(
    r'(?P<subject_id>\d+)'
    r'_(?P<text_id>.+)'
    r'_trial(?P<trial_id>\d+).csv'
)

text_id_codes = [
    'b0', 'b1', 'b2', 'b3', 'b4', 'b5',
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5',
]


def custom_preprocess_potec(
    df_file: pd.DataFrame,
) -> pd.DataFrame:
    df_file.rename(columns={'time': 't'}, inplace=True)
    df_file.t = df_file.t.astype(float)
    df_file.x = df_file.x.astype(float)
    df_file.y = df_file.y.astype(float)

    pos_cols = ['x', 'y']
    df_file[pos_cols] = experiment.screen.pix2deg(
        df_file[pos_cols], center_origin=True)

    return df_file


def read_and_process_potec(
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

    csv_filepaths = get_filepaths(input_rootpath, extension='.csv')
    print('number of csv files:', len(csv_filepaths))
    print(f'{csv_filepaths[0] = }')


    # parse experiment info from filename
    subject_ids = [None for _ in csv_filepaths]
    text_ids = [None for _ in csv_filepaths]
    trial_ids = [None for _ in csv_filepaths]
    for idx, filepath in enumerate(csv_filepaths):

        match = filename_regex.match(filepath.name)

        if match:
            subject_ids[idx] = match.group('subject_id')
            text_ids[idx] = match.group('text_id')
            trial_ids[idx] = match.group('trial_id')

        else:
            raise Exception(f'filepath {filepath} has no parsing match')

    # create dataframe with all fileinfos
    fileinfo_df = pd.DataFrame({
        'subject_id': subject_ids,
        'text_id': text_ids,
        'trial_id': trial_ids,
        'filepath': csv_filepaths,
    })

    fileinfo_df.subject_id = fileinfo_df.subject_id.astype(int)
    fileinfo_df.trial_id = fileinfo_df.trial_id.astype(int)
    fileinfo_df.text_id = fileinfo_df.text_id.apply(text_id_codes.index)
    fileinfo_df.text_id = fileinfo_df.text_id.astype(int)
    fileinfo_df.sort_values(by=['filepath'], inplace=True)

    print("unique subject ids:", fileinfo_df.subject_id.unique())
    print("unique text ids:", fileinfo_df.text_id.unique())
    print("unique trial ids:", sorted(fileinfo_df.trial_id.unique()))
    print(fileinfo_df.head())

    # get pipeline configuration
    pipeline_config = pipelines[pipeline_key]
    print(f'{pipeline_config = }')

    # preprocess files
    X, Y, events, event_properties = preprocess_files(
        fileinfo_df=fileinfo_df,
        X_channels=X_channels,
        Y_columns=Y_columns,
        input_sampling_rate=experiment.sampling_rate,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        pipeline_config=pipeline_config,
        read_csv_kwargs={'delim_whitespace': True, 'na_values': ['.']},
        custom_preprocess=custom_preprocess_potec,
    )

    print('unique subject ids:', np.unique(Y[:, Y_columns['subject_id']]))
    print('unique text ids:', np.unique(Y[:, Y_columns['text_id']]))

    # convert class vector of numbers to binary class matrix
    label_encoder = LabelEncoder()
    label_encoder.fit(Y[:, Y_columns[selected_label]])
    labels = label_encoder.transform(Y[:, Y_columns[selected_label]])

    Y[:, Y_columns[selected_label]] = labels
    Y_cat = to_categorical(labels)
    print(f'{Y_cat.shape = }')

    # create folds with leave-one-text-out scheme:
    # validation set is always test-text-id - 1
    folds = {}
    for test_text_id in sorted(fileinfo_df.text_id.unique()):
        fold_id = test_text_id

        # mod for ring structure
        val_text_id = (test_text_id - 1) % len(text_id_codes)

        mask_test = np.equal(Y[:, Y_columns['text_id']], test_text_id)
        mask_val = np.equal(Y[:, Y_columns['text_id']], val_text_id)
        mask_train = np.logical_not(np.logical_or(mask_test, mask_val))

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
    data = read_and_process_potec(
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
        input_rootpath=basepaths.potec,
        output_rootpath=basepaths.workspace,
         **args,
    )
