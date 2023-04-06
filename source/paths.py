import joblib
import os
import shutil
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Union

from config import Config


class ExperimentPaths:
    """This class holds paths for the experiment."""

    def __init__(
            self,
            config: Config,
            basepath: str = '../',
    ):
        self.base = Path(basepath)

        if config.augmentation is None or config.augmentation.n_augmentations == 0:
            self._data_key = config.dataset.key if config.dataset else None
        else:
            self._data_key = f'{config.dataset.key}_{config.augmentation.key}'

        self._model_key = config.model.key if config.model else None
        self._explainer_key = config.explainer.key if config.explainer else None
        self._metric_key = config.metric.key if config.metric else None
        self._segmentation_key = config.segmentation_key
        self._augmentation_key = config.augmentation.key if config.augmentation else None

        self._paths = {}
        self.setup_repository_paths(self.base, config.repository)
        self.setup_standard_paths(
            basepath=self.base,
            data_key=self._data_key,
            model_key=self._model_key,
        )

    def setup_repository_paths(self, basepath, repository):
        dirpath = repository['dirpath']
        if not dirpath.startswith('/'):
            dirpath = basepath / dirpath
        self.repository = dirpath

        filename_keys = [
            'X', 'X_format', 'Y', 'Y_labels', 'Y_format', 'folds',
            'label_encoder',
        ]

        # join repository filepaths
        for filename_key in filename_keys:
            filename = repository[filename_key]
            self._paths[filename_key] = dirpath / filename

        # join event filepaths
        #event_dirpath = dirpath / repository['event_dirname']
        #self.events = []
        #for event_filename in repository['event_filenames']:
        #    event_filepath = event_dirpath / event_filename
        #    self.events.append(event_filepath)

    def setup_standard_paths(self, basepath: str, data_key: str, model_key: str):
        standard_dirnames = {
            'model': 'models',
            'logs': 'logs',
            'plots': 'plots',
        }
        for dirkey, dirname in standard_dirnames.items():
            dirpath = basepath / dirname / data_key
            if model_key is not None:
                dirpath = dirpath / model_key
            self._paths[dirkey] = dirpath

    def __getitem__(self, key: str):
        return self._paths[key]

    def __setitem__(self, key: str, value: Any):
        self._paths[key] = value

    def __repr__(self):
        return "Paths: " + pformat(vars(self))

    def get_custom_path(
            self,
            dirname: str,
            subdirname: str,
            use_data_key: bool = True,
            use_model_key: bool = True,
            use_explainer_key: bool = True,
            use_metric_key: bool = True,
            use_segmentation_key: bool = False,
    ):
        # basepath for evaluation task name
        eval_path = self.base / dirname / subdirname

        # add data, model, expainer and metric keys if requested
        if self._data_key and use_data_key:
            eval_path = eval_path / self._data_key
        if self._model_key and use_model_key:
            eval_path = eval_path / self._model_key
        if self._explainer_key and use_explainer_key:
            eval_path = eval_path / self._explainer_key
        if self._metric_key and use_metric_key:
            eval_path = eval_path / self._metric_key
        if self._segmentation_key and use_segmentation_key:
            eval_path = eval_path / self._segmentation_key

        return eval_path

    def get_eval_path(
            self, name: str,
            use_data_key: bool = True,
            use_model_key: bool = True,
            use_explainer_key: bool = True,
            use_metric_key: bool = True,
            use_segmentation_key: bool = False,
    ):
        if name == 'model_predictions':
            use_explainer_key = False
            use_metric_key = False
        return self.get_custom_path(
            dirname='evaluations',
            subdirname=name,
            use_data_key=use_data_key,
            use_model_key=use_model_key,
            use_explainer_key=use_explainer_key,
            use_metric_key=use_metric_key,
            use_segmentation_key=use_segmentation_key,
        )

    def get_plot_path(
            self, name: str,
            use_data_key: bool = True,
            use_model_key: bool = True,
            use_explainer_key: bool = True,
            use_metric_key: bool = True,
    ):
        return self.get_custom_path(
            dirname='plots',
            subdirname=name,
            use_data_key=use_data_key,
            use_model_key=use_model_key,
            use_explainer_key=use_explainer_key,
            use_metric_key=use_metric_key,
        )


def makedirs(
        dirpaths: Union[Path, List[Path]],
        clean: bool = False,
        exist_ok: bool = True,
        ignore_errors: bool = True,
) -> None:
    if not isinstance(dirpaths, list):
        dirpaths = [dirpaths]
    for dirpath in dirpaths:
        if clean:
            shutil.rmtree(dirpath, ignore_errors=ignore_errors)
        os.makedirs(dirpath, exist_ok=exist_ok)
