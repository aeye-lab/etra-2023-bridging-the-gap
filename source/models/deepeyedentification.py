from collections import OrderedDict
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy

from .base import Base
from .conv_net import ConvNet
from .conv_net import adam_epsilon_tf_to_torch
from .conv_net import init_weights


class DeepEyedentificationLive(Base):
    def __init__(
            self,
            output_units: int,
            n_channels: int,
            sequence_length: int,
            dense_units: List[int],
            config_fast_subnet,  # TODO: we cannot use Config here as this will
            config_slow_subnet,  #       create a circular import. Fix this.
            path_weights_fast_subnet: Path = None,
            path_weights_slow_subnet: Path = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_func = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
        })

        # get subnet init kwargs
        fast_subnet_init_kwargs = config_fast_subnet.model['init_kwargs']
        slow_subnet_init_kwargs = config_slow_subnet.model['init_kwargs']

        # remove these parameters from init kwargs. we use the passed ones.
        del fast_subnet_init_kwargs['output_units']
        del fast_subnet_init_kwargs['n_channels']
        del fast_subnet_init_kwargs['sequence_length']
        del slow_subnet_init_kwargs['output_units']
        del slow_subnet_init_kwargs['n_channels']
        del slow_subnet_init_kwargs['sequence_length']

        # initialize subnets
        self.fast_subnet = ConvNet(
            output_units=output_units,
            n_channels=n_channels//2,
            sequence_length=sequence_length,
            **fast_subnet_init_kwargs,
        )
        self.slow_subnet = ConvNet(
            output_units=output_units,
            n_channels=n_channels//2,
            sequence_length=sequence_length,
            **slow_subnet_init_kwargs,
        )

        # load pretrained subnet weights if given
        if path_weights_fast_subnet and path_weights_slow_subnet:
            weights_fast_subnet = torch.load(path_weights_fast_subnet)
            weights_slow_subnet = torch.load(path_weights_slow_subnet)
            self.fast_subnet.load_state_dict(weights_fast_subnet)
            self.slow_subnet.load_state_dict(weights_slow_subnet)

        # raise error if only path for a single subnet given to prevent bugs
        elif path_weights_fast_subnet or path_weights_slow_subnet:
            raise ValueError(
                'you need to pass paths to both subnets if you want to load them'
            )

        # get number of output features from fast and slow subnet
        out_features_fast_subnet = fast_subnet_init_kwargs['dense_units'][-1]
        out_features_slow_subnet = slow_subnet_init_kwargs['dense_units'][-1]
        in_features = out_features_fast_subnet + out_features_slow_subnet

        # create layers for dense stack
        fc_layers = []

        # create each block with a dense, relu and batchnorm layer
        for fc_block_id in range(len(dense_units)):
            out_features = dense_units[fc_block_id]
            fc_layers.extend([
                (
                    f'fc_{fc_block_id}',
                    nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                    ),
                ),
                (
                    f'bn_fc_{fc_block_id}',
                    nn.BatchNorm1d(
                        num_features=out_features,
                        momentum=0.01,
                        eps=0.001,
                    ),
                ),
                (
                    f'act_fc_{fc_block_id}',
                    nn.ReLU(),
                ),
            ])
            in_features = out_features

        self.fc_stack = nn.Sequential(OrderedDict([*fc_layers]))

        self.classification_layer = nn.Linear(
            in_features=in_features,
            out_features=output_units,
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, classification: bool = True):
        n_input_channels = x.shape[1]

        if n_input_channels % 2 != 0:
            raise ValueError()

        channel_half = n_input_channels // 2

        # apply fast subnet to first half of the channels
        out_fast = self.fast_subnet(x[:, :channel_half, :], classification=False)

        # apply slow subnet to second half of the channels
        out_slow = self.slow_subnet(x[:, channel_half:, :], classification=False)

        merged_in = torch.cat([out_fast, out_slow], dim=1)

        out = self.fc_stack(merged_in)

        if classification:
            out = self.classification_layer(out)

        return out

    def configure_optimizers(self):
        learning_rate = 0.00011
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon_dash = 1e-07
        epsilon = adam_epsilon_tf_to_torch(
            epsilon_dash=epsilon_dash, beta_2=beta_2,
        )

        optimizer = optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=False,
            amsgrad=False,
        )
        return optimizer
