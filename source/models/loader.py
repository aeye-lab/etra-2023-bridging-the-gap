from typing import Any, Dict

import torch

from config import Config
from paths import ExperimentPaths


def load_model_init_kwargs(
    fold, config, basepath,
) -> Dict[str, Any]:
    # get model setup parameters
    sequence_length = fold.X_train.shape[2]
    n_channels = fold.X_train.shape[1]
    n_classes = fold.Y_train.shape[1]

    # load keyword arguments for model initialization
    model_init_kwargs = config.model['init_kwargs'].copy()

    # replace placeholders with values
    for key, value in model_init_kwargs.items():
        if type(value) == str and value == '$n_classes':
            model_init_kwargs[key] = n_classes
        elif type(value) == str and value == '$n_channels':
            model_init_kwargs[key] = n_channels
        elif type(value) == str and value == '$sequence_length':
            model_init_kwargs[key] = sequence_length
        elif type(value) == str and value == '$mean':
            model_init_kwargs[key] = fold.X_train.mean(axis=(0, 2), keepdims=True)
        elif type(value) == str and value == '$std':
            model_init_kwargs[key] = fold.X_train.std(axis=(0, 2), keepdims=True)
        elif type(value) == str and value == '$mean-mono':
            model_init_kwargs[key] = get_monocular_statistics(fold.X_train, 'mean')
        elif type(value) == str and value == '$std-mono':
            model_init_kwargs[key] = get_monocular_statistics(fold.X_train, 'std')

        # get configs for DEL subnets
        elif key in ['config_fast_subnet', 'config_slow_subnet']:
            subnet_model_key = value
            model_init_kwargs[key] = Config(
                data_key=config.dataset.key, model_key=subnet_model_key,
            )
        # get path to pretrained weights of DEL subnets
        elif key in ['path_weights_fast_subnet', 'path_weights_slow_subnet']:
            # don't load weights if value is None
            if value is None:
                continue
            subnet_model_key = value
            subnet_config = Config(
                data_key=config.dataset.key, model_key=subnet_model_key,
            )
            subnet_paths = ExperimentPaths(config=subnet_config, basepath=basepath)
            subnet_weights_dirpath = subnet_paths['model']
            subnet_weights_filename = f'model_fold_{fold.id}.pth'
            subnet_weights_filepath = subnet_weights_dirpath / subnet_weights_filename
            model_init_kwargs[key] = subnet_weights_filepath

    return model_init_kwargs


def load_model(
        fold, config, filepath, basepath,
) -> torch.nn.Module:
    model_init_kwargs = load_model_init_kwargs(
        fold=fold,
        config=config,
        basepath=basepath,
    )
    print('model_init_kwargs:', model_init_kwargs)

    # initialize model
    model_class = config.model['class']
    model = model_class(**model_init_kwargs)

    # load fitted model state dict
    model.load_state_dict(torch.load(filepath))
    model.eval()

    # move the model to the correct gpu device
    model.to(f'cuda:{config.gpu_id}')
    return model
