import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import seed

from config import Config
from paths import ExperimentPaths
from .loader import load_model_init_kwargs


def train_model(
        fold, config, log_dirpath, random_seed, basepath,
) -> torch.nn.Module:
    # set seed
    seed.seed_everything(random_seed)

    model_init_kwargs = load_model_init_kwargs(
        fold=fold,
        config=config,
        basepath=basepath,
    )
    print('model_init_kwargs:', model_init_kwargs)

    # set custom sampler for training dataloader if requested
    if 'sampler' in config.model:
        sampler_class = config.model['sampler']
        sampler_kwargs = config.model['sampler_kwargs']
        labels = fold.get_labels('test')
        sampler = sampler_class(labels=labels, **sampler_kwargs)
        fold.set_training_sampler(sampler)

    # initialize model
    model_class = config.model['class']
    model = model_class(**model_init_kwargs)

    # perform dry run to initialize the network's lazy modules
    gpu_id = config.gpu_id
    model.to(f'cuda:{gpu_id}')
    batch_size = config.model['batch_size']
    model(torch.ones_like(torch.Tensor(fold.X_train[:batch_size])).to(model.device))

    # initialize logger
    logger = CSVLogger(
        save_dir=log_dirpath, name=None, flush_logs_every_n_steps=1,
    )

    # initialize trainer with early stopping callback and fit model
    if config.model['early_stopping']:
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)
        trainer_callbacks = [early_stop_callback]
    else:
        trainer_callbacks = []

    trainer = Trainer(
        accelerator='gpu',
        devices=[config.gpu_id],
        max_epochs=100,
        enable_progress_bar=True,
        logger=logger,
        callbacks=trainer_callbacks,
    )
    trainer.fit(
        model=model,
        train_dataloaders=fold.train_dl,
        val_dataloaders=fold.val_dl,
    )
    return model, trainer


def create_metric_df_from_pytorch_logger(logger):
    metric_dict = {}
    logged_metrics = logger.experiment.metrics
    for entry in logged_metrics:
        # drop test logs
        if any(key.startswith('test') for key in entry.keys()):
            continue

        # create dict entry on first write
        for metric_name, metric_value in entry.items():
            if metric_name not in metric_dict.keys():
                metric_dict[metric_name] = []

            # epoch and step values should be written only once
            if metric_name in ['epoch', 'step']:
                if len(metric_dict[metric_name]) > 0:
                    last_val = metric_dict[metric_name][-1]
                else:
                    last_val = None
                if last_val == metric_value:
                    continue

            # append value to metric dict
            metric_dict[metric_name].append(metric_value)

    df_metrics = pd.DataFrame(metric_dict)
    df_metrics.set_index('epoch', inplace=True)

    return df_metrics
