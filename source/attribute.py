import argparse
import joblib
import os
import shutil


from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import NoiseTunnel

from config import Config


class Explainer:
    def __init__(
            self,
            model,
            config,
            validation_data,
    ):
        self.model = model
        self.config = config
        self.validation_data = validation_data

        self.initialize_attribute_kwargs(
            config=config,
            validation_data=validation_data,
        )
        self.explainer = self.initialize_explainer(
            model=model,
            config=config,
            validation_data=validation_data,
        )

    def initialize_attribute_kwargs(
            self,
            config,
            validation_data,
    ) -> None:
        # We initialize the attributes in-place.
        attribute_kwargs = config.explainer['attribute_kwargs'].copy()

        # Fill baselines value with aggregates from validation data.
        if 'baseline' in attribute_kwargs:
            if config.explainer['baseline'] == 'min':
                attribute_kwargs['baselines'] = np.nanmin(
                    validation_data)
            elif config.explainer['baseline'] == 'max':
                attribute_kwargs['baselines'] = np.nanmax(
                    validation_data)
            elif config.explainer['baseline'] == 'mean':
                attribute_kwargs['baselines'] = np.nanmean(
                    validation_data)
            elif config.explainer['baseline'] == 'median':
                attribute_kwargs['baselines'] = np.nanmedian(
                    validation_data)
            else:
                raise NotImplementedError(attribute_kwargs['baseline'])

        # TODO: Internal batch size cannot be smaller than current batch size.
        #if 'internal_batch_size': in attribute_kwargs:
        #    attribute_kwargs['internal_batch_size'] = min(
        #        attribute_kwargs['internal_batch_size'],

        # Add model-specific attribute_kwargs to config.explainer
        attribute_kwargs_model = config.model.get('attribute_kwargs', {})

        if config.explainer['framework'] == 'zennit':
            # Zennit doesn't support forwarding args to the model.
            # We need to implement a workaround to let this work.
            if 'additional_forward_args' in attribute_kwargs_model:
                raise NotImplementedError()

        config.explainer['attribute_kwargs'] = {
            **attribute_kwargs, **attribute_kwargs_model,
        }

    def initialize_explainer(
            self,
            model,
            config,
            validation_data,
    ):
        if config.explainer['framework'] == 'captum':
            explainer = self.initialize_captum_explainer(
                explainer_class=config.explainer['class'],
                model=model,
                init_kwargs=config.explainer['init_kwargs'],
                attribute_kwargs=config.explainer['attribute_kwargs'],
            )
        elif config.explainer['framework'] == 'zennit':
            explainer = self.initialize_zennit_explainer(
                explainer_class=config.explainer['class'],
                model=model,
                init_kwargs=config.explainer['init_kwargs'],
                attribute_kwargs=config.explainer['attribute_kwargs'],
            )
        return explainer

    def initialize_captum_explainer(
            self,
            explainer_class,
            model,
            init_kwargs=None,
            attribute_kwargs=None,
    ):
        if attribute_kwargs is None:
            attribute_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        # Initialize explainer.
        explainer = explainer_class(model, **init_kwargs)

        # Wrap NoiseTunnel around if nt_type is specified.
        if 'nt_type' in attribute_kwargs:
            explainer = NoiseTunnel(explainer)

        return explainer

    def initialize_zennit_explainer(
            self,
            explainer_class,
            model,
            init_kwargs=None,
            attribute_kwargs=None,
    ):
        if attribute_kwargs is None:
            attribute_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        # Initialize explainer.
        explainer = explainer_class(**init_kwargs)
        return explainer

    def explain_batch(
            self,
            model,
            inputs,
            targets,
            **kwargs,
    ) -> np.ndarray:
        if model is not None:
            if model != self.model:
                raise ValueError(f'{model} != {self.model}')
        else:
            model = self.model

        if isinstance(targets, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        if isinstance(targets, int) or np.isscalar(targets):
            targets = torch.full(1, fill_value=targets, dtype=int)
        elif isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        inputs = inputs.to(model.device)
        inputs.requires_grad = True
        targets = targets.to(model.device)

        if kwargs:
            raise ValueError(f'{kwargs = }')

        attribute_kwargs = self.config.explainer['attribute_kwargs']

        if self.config.explainer['framework'] == 'captum':
            explain_method = self.explain_batch_captum
        elif self.config.explainer['framework'] == 'zennit':
            explain_method = self.explain_batch_zennit
        else:
            raise ValueError(self.config.explainer['framework'])

        attributions = explain_method(
            explainer=self.explainer,
            model=model,
            inputs=inputs,
            targets=targets,
            attribute_kwargs=attribute_kwargs,
        )
        attributions = attributions.detach().cpu().numpy()

        # take mean of fast and slow subnet attributions in case of del
        if self.config.model['preprocessing'].startswith('del-merged'):
            channel_half = attributions.shape[1] // 2
            attributions = (attributions[:, :channel_half] +
                            attributions[:, channel_half:]) / 2

        # For now this is only used internally for quantus.  We have
        # to take mean across channel axis, as quantus doesn't fully
        # support multi-channel attributions.
        attributions = np.mean(attributions, axis=1, keepdims=True)

        return attributions

    def explain_batch_captum(
        self,
        explainer,
        model,
        inputs,
        targets,
        attribute_kwargs,
        progressbar=True,
        **kwargs,
    ) -> np.ndarray:
        attributions = explainer.attribute(
            inputs=inputs,
            target=targets,
            **attribute_kwargs,
        )
        return attributions

    def explain_batch_zennit(
        self,
        explainer,
        model,
        inputs,
        targets,
        attribute_kwargs,
        progressbar=True,
        **kwargs,
    ) -> np.ndarray:
        with explainer.context(model) as explainer_model:
            # select predicted class as gradient source
            outputs = explainer_model(inputs)
            n_classes = outputs.shape[1]
            grad_outputs = torch.eye(n_classes)[targets].to(model.device)
            attributions, = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
            )
        return attributions


def compute_attributions(
        explainer_config,
        model_config,
        model,
        dataloader,
        val_data,
        predictions,
        progressbar=True,
):
    # compute baseline value
    if 'baseline' in explainer_config:
        if explainer_config['baseline'] == 'min':
            explainer_config['attribute_kwargs']['baselines'] = np.nanmin(val_data)
        elif explainer_config['baseline'] == 'max':
            explainer_config['attribute_kwargs']['baselines'] = np.nanmax(val_data)
        elif explainer_config['baseline'] == 'mean':
            explainer_config['attribute_kwargs']['baselines'] = np.nanmean(val_data)
        elif explainer_config['baseline'] == 'median':
            explainer_config['attribute_kwargs']['baselines'] = np.nanmedian(val_data)
        else:
            raise NotImplementedError(explainer_config['baseline'])

    if explainer_config['framework'] == 'captum':
        attribute_kwargs_e = explainer_config['attribute_kwargs']
        attribute_kwargs_m = model_config.get('attribute_kwargs', {})
        attribute_kwargs = {**attribute_kwargs_e, **attribute_kwargs_m}

        attributions = compute_captum_attributions(
            explainer_class=explainer_config['class'],
            init_kwargs=explainer_config['init_kwargs'],
            attribute_kwargs=attribute_kwargs,
            model=model,
            dataloader=dataloader,
            predictions=predictions,
            noise_tunnel=explainer_config.get('noise_tunnel', False),
            progressbar=progressbar,
        )
    elif explainer_config['framework'] == 'zennit':
        attribute_kwargs_e = explainer_config['attribute_kwargs']
        attribute_kwargs_m = model_config.get('attribute_kwargs', {})

        # zennit doesn't support this. we need a workaround
        if 'additional_foward_args' in attribute_kwargs_m:
            raise NotImplementedError()

        attributions = compute_zennit_attributions(
            explainer_class=explainer_config['class'],
            init_kwargs=explainer_config['init_kwargs'],
            attribute_kwargs=explainer_config['attribute_kwargs'],
            model=model,
            dataloader=dataloader,
            predictions=predictions,
            progressbar=progressbar,
        )

    # take mean of fast and slow subnet attributions in case of del
    if model_config['preprocessing'].startswith('del-merged'):
        channel_half = attributions.shape[1] // 2
        attributions = (attributions[:, :channel_half] +
                        attributions[:, channel_half:]) / 2

    return attributions

def compute_captum_attributions(
        explainer_class,
        attribute_kwargs,
        model,
        dataloader,
        predictions,
        init_kwargs=None,
        noise_tunnel=False,
        progressbar=True,
):
    # initialize attribution object
    if init_kwargs is None:
        init_kwargs = {}
    explainer = explainer_class(model, **init_kwargs)

    if noise_tunnel:
        explainer = NoiseTunnel(explainer)

    # initialize fold attributions
    attributions = np.empty(dataloader.dataset.tensors[0].shape)
    attributions[:, :] = np.nan

    # compute attributions
    b_start_idx = 0
    for xb, _ in tqdm(dataloader, desc='Attributing', disable=not progressbar):
        this_batch_size = len(xb)
        b_end_idx = b_start_idx + this_batch_size

        batch_predictions = predictions[b_start_idx:b_end_idx]
        batch_predictions = list(batch_predictions)
        batch_predictions = [int(c) for c in batch_predictions]

        attr_batch = explainer.attribute(
            inputs=xb.requires_grad_(True).to(model.device),
            target=list(batch_predictions),
            **attribute_kwargs,
        )

        attributions[b_start_idx:b_end_idx] = attr_batch.detach().cpu().numpy()
        b_start_idx = b_end_idx

    return attributions


def one_hot_max(outputs):
    n_classes = outputs.shape[1]
    return torch.eye(n_classes)[torch.argmax(outputs, axis=1)]


def compute_zennit_attributions(
        explainer_class,
        init_kwargs,
        attribute_kwargs,
        model,
        dataloader,
        predictions,
        noise_tunnel=False,
        progressbar=True,
):
    if noise_tunnel:
        raise NotImplementedError("noise tunnel not integrated yet for zennit")

    # initialize attribution object
    composite = explainer_class(**init_kwargs)

    # initialize fold attributions
    attributions = np.empty(dataloader.dataset.tensors[0].shape)
    attributions[:, :] = np.nan

    # compute attributions
    with composite.context(model) as explainer_model:
        b_start_idx = 0
        for xb, _ in tqdm(dataloader, desc='Attributing', disable=not progressbar):
            this_batch_size = len(xb)
            b_end_idx = b_start_idx + this_batch_size


            #xb.requires_grad = True
            #output, relevance = explainer(xb.to(model.device), one_hot_max)

            # select predicted class as gradient source
            inputs = xb.to(model.device)
            inputs.requires_grad = True
            outputs = explainer_model(inputs)
            grad_outputs = one_hot_max(outputs).to(model.device)

            attribution, = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
            )

            attributions[b_start_idx:b_end_idx] = attribution.cpu()
            b_start_idx = b_end_idx

    return attributions
