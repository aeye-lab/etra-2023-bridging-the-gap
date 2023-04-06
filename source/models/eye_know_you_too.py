# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Authors: Dillon Lohr (djl70@txstate.edu)
# Modified by anonymous


from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from torch import nn
from torch import optim
from torchmetrics import Accuracy

from .base import Base


def init_weights(modules_list: Union[nn.Module, List[nn.Module]]) -> None:
    if not isinstance(modules_list, List):
        modules_list = [modules_list]

    for m in modules_list:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)


class ConvBlock(nn.Module):
    """BatchNorm1d + ReLU + Conv1d + optional dense connection """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_block_id: int,
        dilation: int = 1,
        add_dense_connection: bool = True,
        skip_bn_relu: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.add_dense_connection = add_dense_connection

        layers = []

        if not skip_bn_relu:
            layers.extend([
                (
                    f'bn_conv_{conv_block_id}',
                    nn.BatchNorm1d(input_channels),
                ),
                (
                    f'act_conv_{conv_block_id}',
                    nn.ReLU(inplace=False),
                ),
            ])

        layers.append(
            (
                f'conv_{conv_block_id}',
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding='same',
                    dilation=dilation,
                    bias=False,
                ),
            ),
        )
        self.block = nn.Sequential(OrderedDict([*layers]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)

        if self.add_dense_connection:
            return torch.cat([x, out], dim=1)
        else:
            return out


class ConvStack(nn.Module):
    """Series of convolution blocks with optional dense connections."""

    def __init__(
        self,
        n_layers: int,
        input_channels: int,
        growth_rate: int,
        max_dilation: int = 64,
        add_dense_connections: bool = True,
        skip_bn_relu_first_layer: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()

        dilation_exp_mod = int(np.log2(max_dilation)) + 1

        def dilation_at_i(i: int) -> int:
            return 2 ** (i % dilation_exp_mod)

        if add_dense_connections:
            # each conv layer increases input channels by additive growth rate
            layer_input_channels = [
                input_channels + i * growth_rate
                for i in range(n_layers)
            ]
        else:
            # first conv layer gets number of input channels
            # all next layers only get static value of growth rate as input channels
            layer_input_channels = [input_channels] + [
                growth_rate for _ in range(n_layers - 1)
            ]

        conv_blocks = [
            (
                f'block_conv_{conv_block_id}',
                ConvBlock(
                    input_channels=layer_input_channels[conv_block_id],
                    output_channels=growth_rate,
                    dilation=dilation_at_i(conv_block_id),
                    add_dense_connection=add_dense_connections,
                    skip_bn_relu=(conv_block_id == 0 and skip_bn_relu_first_layer),
                    kernel_size=kernel_size,
                    conv_block_id=conv_block_id,
                ),
            )
            for conv_block_id in range(n_layers)
        ]
        self.stack = nn.Sequential(OrderedDict([*conv_blocks]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        return x


class ClassifierBlock(nn.Module):
    """Optional classification block"""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(input_dim, n_classes)

        init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x



class EKY2(Base):
    """
    Network with a single dense block.

    References
    ----------
    https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
    """

    def __init__(
        self,
        input_channels: int,
        output_units: int,
        embedding_dim: int = 128,
        depth: int = 8,
        growth_rate: int = 32,
        max_dilation: int = 64,
        kernel_size: int = 3,
        add_dense_connections: bool = True,
        loss_weights_categorical: float = 0.1,
        loss_weights_metric: float = 1.0,
        default_forward_mode: str = 'embedding',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.categorical_loss_func = nn.CrossEntropyLoss()
        self.metric_loss_miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss_func = losses.MultiSimilarityLoss(
            alpha=2, beta=50, base=0.5,
        )
        self.loss_weights = {
            'categorical': loss_weights_categorical,
            'metric': loss_weights_metric,
        }

        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
        })

        self.default_forward_mode = default_forward_mode

        n_fixed_layers = 1  # embedding layer
        n_layers_per_block = depth - n_fixed_layers
        assert n_layers_per_block > 0, "`depth` is too small"

        # All conv blocks as a single stack
        self.conv_stack = ConvStack(
            n_layers_per_block,
            input_channels,
            growth_rate,
            max_dilation=max_dilation,
            add_dense_connections=add_dense_connections,
            skip_bn_relu_first_layer=True,
            kernel_size=kernel_size,
        )

        if add_dense_connections:
            # each conv layer increases input channels by additive growth rate
            conv_stack_output_channels = input_channels + n_layers_per_block * growth_rate
        else:
            # first conv layer gets number of input channels
            # all next layers only get static value of growth rate as input channels
            conv_stack_output_channels = growth_rate

        # Global average pooling and embedding layer
        self.global_pooling = nn.Sequential(OrderedDict([
            ('global_bn', nn.BatchNorm1d(conv_stack_output_channels)),
            ('global_relu', nn.ReLU(inplace=False)),
            ('global_pool', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten()),
        ]))

        self.fc_embed = nn.Linear(conv_stack_output_channels, embedding_dim)
        self.fc_class = ClassifierBlock(embedding_dim, output_units)

        # Initialize weights
        init_weights(self.modules())

    def forward(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv_stack(x)
        x = self.global_pooling(x)

        if mode is None:
            mode = self.default_forward_mode

        if mode == 'class':
            x = self.fc_embed(x)
            return self.fc_class(x)
        elif mode == 'embedding':
            return self.fc_embed(x)
        elif mode == 'both':
            embedding = self.fc_embed(x)
            class_out = self.fc_class(embedding)
            return class_out, embedding
        else:
            raise ValueError(f'unsupported mode `{mode}`')

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.0001,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=1,
            epochs=100,
        )
        return [optimizer], [scheduler]

    def any_step(self, batch, batch_idx, set_prefix: str):
        xb, yb = batch
        labels = torch.argmax(yb, dim=1)
        yb_pred, embeddings = self(xb, mode='both')

        miner_output = self.metric_loss_miner(embeddings, labels)
        metric_loss = self.metric_loss_func(embeddings, labels, miner_output)
        weighted_metric_loss = metric_loss * self.loss_weights['metric']

        cat_loss = self.categorical_loss_func(yb_pred, yb)
        weighted_cat_loss = cat_loss * self.loss_weights['categorical']

        loss = weighted_metric_loss + weighted_cat_loss

        outputs = {
            f'{set_prefix}_loss': loss.detach(),
            f'{set_prefix}_metric_loss': metric_loss.detach(),
            f'{set_prefix}_categorical_loss': cat_loss.detach(),
        }

        if set_prefix == 'train':
            outputs['loss'] = loss

        yb_uncat = torch.argmax(yb.squeeze(), dim=1)
        for metric_name, metric_func in self.metrics.items():
            metric_value = metric_func(yb_pred, yb_uncat).detach()
            outputs[f'{set_prefix}_{metric_name}'] = metric_value

        return outputs
