import os
from copy import deepcopy
from pprint import pformat
from typing import Any, Dict, Optional

from config.datasets import datasets
from config.explainers import explainers
from config.models import models
from config.metrics import metrics
from config.repositories import repositories
from experiment import ExperimentKeys


class BaseConfig:
    """This is the base class for specific configuration classes."""

    def __init__(self, key, **kwargs):
        if key not in self._definitions.keys():
            raise ValueError(
                f'Unknown {self.__class__.__name__} key \'{key}\'.\n'
                f'Valid keys are: {list(self._definitions.keys())}'
            )
        self.key = key
        self.load_definitions(key)

    def load_definitions(self, key):
        loaded_definitions = deepcopy(self._definitions[key])
        self.__dict__.update(loaded_definitions)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return self.key

    def __str__(self):
        return f"{self.__class__.__name__}: " + pformat(vars(self))

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default


class DataSetConfig(BaseConfig):
    """This class holds configuration data for experiment data sets."""
    _definitions = datasets

    @classmethod
    def keymap(cls, key: str):
        if key in ['mnist1d', 'mnist1d-1000']:
            key = 'mnist1d_sl1000'

        elif key == 'gazebase-1000':
            key = 'gazebase_all_sr1000_sl1000_dxy'

        elif key in ['gazebase', 'gazebase-5000']:
            key = 'gazebase_all_sr1000_sl5000_dxy'

        elif key in ['judo', 'judo-bino', 'judo-1000-bino']:
            key = 'judo_sr1000_sl1000_bxy'

        elif key in ['judo-mono', 'judo-1000-mono']:
            key = 'judo_sr1000_sl1000_rxy'

        elif key in ['potec', 'potec-1000']:
            key = 'potec_sr1000_sl1000_dxy'

        return key


class ModelConfig(BaseConfig):
    """This class holds configuration data for experiment models."""
    _definitions = models

    def __init__(self, key):
        super().__init__(key)

        if 'init_kwargs' not in self.__dict__:
            self.init_kwargs = {}


class ExplainerConfig(BaseConfig):
    """This class holds configuration data for experiment explainers."""
    _definitions = explainers

    def __init__(self, key):
        super().__init__(key)

        if 'init_kwargs' not in self.__dict__:
            self.init_kwargs = {}
        if 'attribute_kwargs' not in self.__dict__:
            self.attribute_kwargs = {}


class MetricConfig(BaseConfig):
    """This class holds configuration data for experiment metrics."""
    _definitions = metrics

    def __init__(self, key):
        super().__init__(key)

        if 'init_kwargs' not in self.__dict__:
            self.init_kwargs = {}
        if 'call_kwargs' not in self.__dict__:
            self.call_kwargs = {}


class AugmentationConfig(BaseConfig):
    """
    This class holds config information for augmentations.
    """
    _key_format = 'naugs{n_augmentations}_std{noise_std}'

    def __init__(
        self,
        key: Optional[str] = None,
        n_augmentations: Optional[int] = None,
        noise_std: Optional[float] = None,
    ):
        self.n_augmentations = n_augmentations
        self.noise_std = noise_std
        # generate key if None
        self.key = key if key else self.generate_key()

    def generate_key(self):
        return self._key_format.format(
            n_augmentations=self.n_augmentations,
            noise_std=self.noise_std,
        )


class Config:
    def __init__(
            self,  # TODO: use ExperimentKey and get rid of key arguments
            experiment_keys: Optional[ExperimentKeys] = None,
            data_key: Optional[str] = None,
            model_key: Optional[str] = None,
            explainer_key: Optional[str] = None,
            metric_key: Optional[str] = None,
            segmentation_key: Optional[str] = None,
            augmentation_kwargs: Optional[Dict[str, Any]] = None,
            gpu_id: int = 0,
            n_workers: int = 4,
            basepath: str = '../',
    ):
        if experiment_keys is not None:
            self.keys = experiment_keys
            if data_key is not None:
                raise ValueError(
                    "data_key must be None if passing experiment_keys"
                )
            if model_key is not None:
                raise ValueError(
                    "model_key must be None if passing experiment_keys"
                )
            if explainer_key is not None:
                raise ValueError(
                    "explainer_key must be None if passing experiment_keys"
                )
            if metric_key is not None:
                raise ValueError(
                    "metric_key must be None if passing experiment_keys"
                )
            if augmentation_kwargs is not None:
                raise ValueError(
                    "augmentation_kwargs must be None if passing experiment_keys"
                )

            # checks are handled. now get parameters from experiment_keys
            data_key = experiment_keys.data
            model_key = experiment_keys.model
            explainer_key = experiment_keys.explainer
            metric_key = experiment_keys.metric
            segmentation_key = experiment_keys.segmentation

            augmentation_kwargs = {
                'n_augmentations': experiment_keys.n_augmentations,
                'noise_std': experiment_keys.noise_std,
            }
        else:
            if augmentation_kwargs is None:
                augmentation_kwargs = {}

            self.keys = ExperimentKeys(
                data_key=data_key,
                model_key=model_key,
                explainer_key=explainer_key,
                metric_key=metric_key,
                segmentation_key=segmentation_key,
                **augmentation_kwargs,
            )

        if self.keys.data is None:
            self.dataset = None
            self.repository = None
        else:
            self.dataset = DataSetConfig(self.keys.data)
            self.repository = deepcopy(repositories[self.dataset.repo_key])

        if self.keys.model is None:
            self.model = None
        else:
            self.model = ModelConfig(self.keys.model)

        if self.keys.explainer is None:
            self.explainer = None
        else:
            self.explainer = ExplainerConfig(self.keys.explainer)

        if self.keys.metric is None:
            self.metric = None
        else:
            self.metric = MetricConfig(self.keys.metric)

        self.segmentation_key = segmentation_key

        self.augmentation = AugmentationConfig(
            n_augmentations=self.keys.n_augmentations,
            noise_std=self.keys.noise_std,
        )

        self.gpu_id = gpu_id
        self.n_workers = n_workers
