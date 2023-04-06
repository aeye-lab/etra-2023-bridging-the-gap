import joblib
import json
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from paths import ExperimentPaths, makedirs


def write_data(
    data: Dict[str, Any],
    data_key: str,
    basepath: str,
    additional_labels: Optional[List[str]] = None,
) -> None:
    # load config and make repository path
    config = Config(
        data_key=data_key,
    )
    paths = ExperimentPaths(
        config=config,
        basepath=basepath,
    )

    print(f'{paths.repository = }')
    makedirs(paths.repository, exist_ok=True)

    # save data
    np.save(paths['X'], data['X'], allow_pickle=False)
    np.save(paths['Y_labels'], data['Y_labels'], allow_pickle=False)
    np.save(paths['Y'], data['Y_cat'], allow_pickle=False)

    with open(paths['X_format'], 'w') as f:
        json.dump(data['X_format'], f)
    with open(paths['Y_format'], 'w') as f:
        json.dump(data['Y_format'], f)

    if additional_labels:
        for label in additional_labels:
            np.save(paths.repository / f'Y_{label}.npy', data[f'Y_{label}'], allow_pickle=False)
            joblib.dump(data[f'{label}_encoder'], paths.repository / f'{label}_encoder.joblib')

    joblib.dump(data['folds'], paths['folds'])
    joblib.dump(data['label_encoder'], paths['label_encoder'])

    events_dirpath = paths.repository / 'events'
    print(f'{events_dirpath = }')
    makedirs(events_dirpath, exist_ok=True)

    for event_name, event_arr in data['events'].items():
        np.save(events_dirpath / f'{event_name}.npy', event_arr, allow_pickle=False)

    for event_name, event_df in data['event_properties'].items():
        event_df.to_csv(events_dirpath / f'{event_name}.csv')

    lscmd = ['ls', '-lhR', paths.repository]
    print(subprocess.check_output(lscmd).decode('utf-8'))
