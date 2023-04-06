from __future__ import annotations

import itertools
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from pymovements.transforms import cut_into_subsequences
from pymovements.transforms import downsample
from pymovements.transforms import pos2vel
from pymovements.transforms import vnorm
from tqdm.auto import tqdm

from event_detection import detect_events
from event_detection import align_segmentation
from event_detection import align_df


class PreprocessedFile:

    def __init__(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            events: dict[str, np.ndarray],
            event_properties: dict[str, pd.DataFrame],
            info: pd.DataFrame,
    ):
        self.X = X
        self.Y = Y
        self.events = events
        self.event_properties = event_properties
        self.info = info


def preprocess_files(
        fileinfo_df: pd.DataFrame,
        X_channels: dict[str, int],
        Y_columns: dict[str, int],
        input_sampling_rate: int,
        sequence_length: int,
        downsampling_factor: int,
        pipeline_config: dict[str, Any],
        csv_groupby: Union[str, list[str]] = None,
        read_csv_kwargs: Optional[dict[str, Any]] = None,
        custom_preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    # read and preprocess input files
    preprocessed_eyegaze_data = []
    for ix_df, fileinfo in tqdm(fileinfo_df.iterrows(), total=len(fileinfo_df)):
        preprocessed_file = preprocess_file(
            fileinfo=fileinfo,
            X_channels=X_channels,
            Y_columns=Y_columns,
            input_sampling_rate=input_sampling_rate,
            sequence_length=sequence_length,
            downsampling_factor=downsampling_factor,
            pipeline_config=pipeline_config,
            csv_groupby=csv_groupby,
            read_csv_kwargs=read_csv_kwargs,
            custom_preprocess=custom_preprocess,
        )
        preprocessed_eyegaze_data.append(preprocessed_file)

    # create big X and Y for all instances to be written
    n_instances = np.sum([
        preprocessed_file.X.shape[0]
        for preprocessed_file in preprocessed_eyegaze_data
    ])
    print(f'{n_instances = }')

    X = np.zeros((n_instances, *preprocessed_file.X.shape[1:]))
    Y = np.zeros((n_instances, len(Y_columns)))
    print(f'{X.shape = }')
    print(f'{Y.shape = }')

    events = {key: np.zeros((n_instances, sequence_length), dtype=bool)
              for key in preprocessed_file.events.keys()}
    print(f'events = {list(events.keys())}')

    event_properties = {
        key: pd.DataFrame()
        for key in preprocessed_file.event_properties.keys()
    }

    # fill X and Y with preprocessed data and fileinfo
    last_instance_id = 0
    for file_id, preprocessed_file in enumerate(tqdm(preprocessed_eyegaze_data)):
        n_instances_in_file = preprocessed_file.X.shape[0]
        first_instance_id = last_instance_id
        last_instance_id = first_instance_id + n_instances_in_file
        instance_slice = slice(first_instance_id, last_instance_id)

        X[instance_slice] = preprocessed_file.X
        Y[instance_slice] = preprocessed_file.Y

        for event_name, event_arr in preprocessed_file.events.items():
            events[event_name][instance_slice] = event_arr

        file_instance_ids = range(first_instance_id, last_instance_id)

        for event_name, event_dfs in preprocessed_file.event_properties.items():
            for event_df, instance_id in zip(event_dfs, file_instance_ids):
                event_df.insert(0, 'instance_id', instance_id)

                # TODO: this is inefficient and takes a long time
                event_properties[event_name] = pd.concat(
                    [event_properties[event_name], event_df],
                    ignore_index=True,
                )

    return X, Y, events, event_properties


def preprocess_file(
    fileinfo: pd.DataFrame,
    X_channels: dict[str, int],
    Y_columns: dict[str, int],
    input_sampling_rate: int,
    sequence_length: int,
    downsampling_factor: int,
    pipeline_config: dict[str, Any],
    csv_groupby: Union[str, list[str]] = None,
    read_csv_kwargs: Optional[dict[str, Any]] = None,
    custom_preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> PreprocessedFile:

    df_file = pd.read_csv(fileinfo.filepath, **read_csv_kwargs)

    if custom_preprocess is not None:
        df_file = custom_preprocess(df_file)

    preprocessed_result = preprocess_dataframe(
        df=df_file,
        X_channels=X_channels,
        sampling_rate=input_sampling_rate,
        sequence_length=sequence_length,
        downsampling_factor=downsampling_factor,
        groupby=csv_groupby,
        **pipeline_config,
    )

    if csv_groupby is None:
        X_file, events_file, event_properties_file = preprocessed_result
    else:
        X_file, Y_groups_file, events_file, event_properties_file = preprocessed_result

    Y_file = np.zeros((X_file.shape[0], len(Y_columns))) * np.nan

    if csv_groupby is not None:
        for group_name in Y_groups_file.columns:
            if group_name in Y_columns.keys():
                Y_file[:, Y_columns[group_name]] = Y_groups_file[group_name]

    for fileinfo_column in fileinfo.index:
        if fileinfo_column in Y_columns.keys():
            Y_file[:, Y_columns[fileinfo_column]] = fileinfo[fileinfo_column]

    preprocessed_file = PreprocessedFile(
        X=X_file,
        Y=Y_file,
        events=events_file,
        event_properties=event_properties_file,
        info=fileinfo,
    )

    return preprocessed_file


def preprocess_dataframe(
        df: pd.DataFrame,
        X_channels: dict[str, int],
        sequence_length: int,
        sampling_rate: float,
        pos2vel_method: str,
        pos2vel_kwargs: dict[str, Any],
        max_velocity: float,
        nan_threshold: float,
        downsampling_factor: int = 1,
        groupby: Union[str, list[str]] = None,
) -> np.ndarray:
    '''
    We expect the dataframe to have the columns ['t', 'x', 'y'] for
    monocular data and ['t', 'xr', 'yr', 'xl', 'xr'] for binocluar data.
    '''
    # TODO: make parameter for that
    event_detection_methods = ['engbert']

    if groupby is not None:
        if type(groupby) == str:
            groupby = [groupby]
        X_groups = {}
        properties_groups = {}
        events_groups = {}
        # apply recursively on each group
        for group, df_group in df.groupby(groupby):
            X_groups[group], events_groups[group], properties_groups[group] = preprocess_dataframe(
                df=df_group,
                X_channels=X_channels,
                sequence_length=sequence_length,
                sampling_rate=sampling_rate,
                downsampling_factor=downsampling_factor,
                pos2vel_method=pos2vel_method,
                pos2vel_kwargs=pos2vel_kwargs,
                max_velocity=max_velocity,
                nan_threshold=nan_threshold,
            )

        group_names = list(X_groups.keys())
        n_instances = sum([X.shape[0] for X in X_groups.values()])
        n_channels = X_groups[group].shape[2]

        # unpack X from each group and concatenate
        X = np.concatenate([*X_groups.values()])

        # now fill Y with groups from groupby
        Y = pd.DataFrame(columns=groupby, index=range(n_instances))
        last_instance_id = 0
        for group_name in group_names:
            X_group = X_groups[group_name]
            n_instances_group = X_group.shape[0]
            first_instance_id = last_instance_id
            last_instance_id = first_instance_id + n_instances_group
            group_slice = slice(first_instance_id, last_instance_id)
            Y[group_slice] = group_name

        # merge event dict items by concatenating arrays
        events = {}
        for event_name in events_groups[group_names[0]].keys():
            # we create a list of arrays first
            events[event_name] = [None for _ in group_names]
            for group_id, group_name in enumerate(group_names):
                events[event_name][group_id] = events_groups[group_name][event_name]


            # now we concatenate the arrays
            events[event_name] = np.concatenate(events[event_name])

        properties = {}
        for event_name in properties_groups[group_names[0]].keys():
            # we create a list of dataframes first
            properties[event_name] = [None for _ in group_names]
            for group_id, group_name in enumerate(group_names):
                properties[event_name][group_id] = properties_groups[group_name][event_name]

            # now we concatenate the dataframes
            properties[event_name] = [e for es in properties[event_name] for e in es]

        return X, Y, events, properties

    monocular_pos_cols = ['x', 'y']
    binocular_pos_cols = ['xr', 'yr', 'xl', 'xl']

    if 't' not in df.columns:
        raise ValueError(f"time column 't' is missing. columns: {df.columns}")
    if pd.Series(monocular_pos_cols).isin(df.columns).all():
        is_monocular = True
        pos_cols = monocular_pos_cols
    elif pd.Series(binocular_pos_cols).isin(df.columns).all():
        is_monocular = False
        pos_cols = binocular_pos_cols
    else:
        raise ValueError(
            f"could not determine if data is monocular. columns: {df.columns}"
        )

    if downsampling_factor != 1:
        df = downsample(df, downsampling_factor)

    # convert positional data to velocities
    if is_monocular:
        vel_channels = [X_channels['dx'], X_channels['dy']]

        df.loc[:, ['dx', 'dy']] = pos2vel(
            df[['x', 'y']].values,
            sampling_rate=sampling_rate,
            method=pos2vel_method,
            **pos2vel_kwargs,
        )
    else:
        vel_channels = [
            X_channels['dxr'], X_channels['dyr'],
            X_channels['dxl'], X_channels['dyl'],
        ]

        df.loc[:, ['dxr', 'dyr']] = pos2vel(
            df[['xr', 'yr']].values,
            sampling_rate=sampling_rate,
            method=pos2vel_method,
            **pos2vel_kwargs,
        )
        df.loc[:, ['dxl', 'dyl']] = pos2vel(
            df[['xl', 'yl']].values,
            sampling_rate=sampling_rate,
            method=pos2vel_method,
            **pos2vel_kwargs,
        )

    # create initial X_uncut array with format spec from X_channels dict
    X_uncut = np.ones((1, len(df), len(X_channels)))
    for df_column_name, X_channel_id in X_channels.items():
        X_uncut[:, :, X_channel_id] = df[df_column_name]
    events_uncut = {}
    event_properties_uncut = {}

    # Clamp velocities to maximum velocity (on channel by channel basis)
    v = X_uncut[:, :, vel_channels]
    events_uncut['clip'] = np.logical_or(v < -max_velocity, v > max_velocity).any(axis=2)[0]
    X_uncut[:, :, vel_channels] = np.clip(v, -max_velocity, max_velocity)

    # TODO: implement rescaling
    # this won't do for absolute velocity clipping, we would need to scale v
    #X[:, :, vel_channels] = np.where(vnorm(v, axis=2) > max_vel, max_vel, v)

    # we apply the event detection methods on the uncut data
    for event_detection_method in event_detection_methods:
        if event_detection_method != 'engbert':
            raise NotImplementedError(event_detection_method)

        event_properties, events_method = detect_events(
            method=event_detection_method,
            df=df,
            is_monocular=is_monocular,
        )
        events_uncut = {**events_uncut, **events_method}
        event_properties_uncut = {**event_properties_uncut, **event_properties}

    # cut long sequence into smaller subsequences
    X = cut_into_subsequences(
        arr=X_uncut,
        window_size=sequence_length,
        keep_padded=False,
    )

    # align events to subsequences
    events = {}
    for event_key, event_segmentation_uncut in events_uncut.items():
        aligned_segmentation = align_segmentation(
            t_uncut=X_uncut[:, :, X_channels['t']],
            t_cut=X[:, :, X_channels['t']],
            segmentation=event_segmentation_uncut,
        )
        events[event_key] = aligned_segmentation

    event_properties = {}
    for event_key, df_event_properties in event_properties_uncut.items():
        df_aligned = align_df(
            t_uncut=X_uncut[:, :, X_channels['t']],
            t_cut=X[:, :, X_channels['t']],
            df=df_event_properties,
        )
        event_properties[event_key] = df_aligned

    # exclude sequences with more than 50% NaN in velocity channels (take sum)
    X_vel = X[:, :, vel_channels]
    events['nan'] = np.isnan(X_vel).any(axis=2)
    n_nans = np.isnan(X_vel).any(axis=2).sum(axis=1)
    valid_sequence_mask = n_nans < nan_threshold * sequence_length

    # select only valid sequence ids
    X = X[valid_sequence_mask]
    for event_name in events.keys():
        events[event_name] = events[event_name][valid_sequence_mask]
        if event_name in ['engbert.saccade', 'ivt.fixation']:
            event_properties[event_name] = list(itertools.compress(
                event_properties[event_name], valid_sequence_mask
            ))

    # Get segmentation for all timesteps that don't belong to events.
    events['unclassified'] = np.sum(list(events.values()), axis=0) == 0

    return X, events, event_properties
