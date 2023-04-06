from collections import OrderedDict
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy

import pytorch_lightning as pl
import torchvision

from .base import Base


@torch.no_grad()
def keras_he_normal_(tensor: torch.Tensor) -> None:
    # Keras uses a truncated normal distribution but PyTorch does not.
    # So we need to use a different initialization for the weights than
    # `nn.init.kaiming_normal_`.  The truncated normal distribution used
    # by Keras resamples values that are more than two standard
    # deviations from the mean.
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode="fan_in")
    std = np.sqrt(2.0 / fan)
    torch.nn.init.trunc_normal_(
        tensor,
        mean=0.0, std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )


def init_weights(modules_list: Union[nn.Module, List[nn.Module]]) -> None:
    if not isinstance(modules_list, List):
        modules_list = [modules_list]

    for m in modules_list:
        if isinstance(m, nn.Conv1d):
            keras_he_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def adam_epsilon_tf_to_torch(epsilon_dash: float, beta_2: float) -> float:
    # tensorflow actually uses epsilon' as epsilon for adam optimizer.
    # as we take this parameter from the tf implementation we need to
    # convert it.
    # https://stackoverflow.com/questions/57824804/epsilon-parameter-in-adam-opitmizer
    epsilon = epsilon_dash / np.sqrt(1 - beta_2)
    return epsilon


class ConvBlock(nn.Module):
    """BatchNorm1d + ReLU + Conv1d + optional dense connection """

    def __init__(
        self,
        conv_block_id: int,
        in_channels: int,
        out_channels: int,
        conv_size: int,
        conv_stride: int,
        conv_padding: int,
        conv_padding_mode: str,
        pool_size: int,
        pool_stride: int,
        pool_padding: str,
        pool_before_bn: bool,
    ):
        super().__init__()

        layers = []

        layers.extend([
            (
                f'conv_{conv_block_id}',
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    padding_mode=conv_padding_mode,
                ),
            ),
            (
                f'act_conv_{conv_block_id}',
                nn.ReLU(),
            ),
            (
                f'bn_conv_{conv_block_id}',
                nn.BatchNorm1d(
                    num_features=out_channels,
                    momentum=0.01,
                    eps=0.001,
                ),
            ),
            (
                f'pool_conv_{conv_block_id}',
                nn.AvgPool1d(
                    kernel_size=pool_size,
                    stride=pool_stride,
                ),
            ),
        ])

        layer_idx_pool = 3

        if pool_before_bn:
            idx_old = layer_idx_pool
            idx_new = layer_idx_pool - 1
            layers[idx_new], layers[idx_old] = layers[idx_old], layers[idx_new]
            layer_idx_pool = idx_new

        if pool_padding == 'same':
            # The original Keras implementation uses AveragePooling1D with
            # padding = "same".  When the kernel size is 2 and the stride is
            # 1, we would need to pad only one side of the input.  This is
            # not supported by PyTorch.  Instead, we add a padding layer
            # before the pooling layer to accomplish this.  Additionally,
            # according to an example in the Keras documentation, the
            # padding needs to be done on the right side of the input in
            # this case, and the padding must not be counted in the average
            # pooling.  We accomplish this by using replication padding
            # instead of zero padding.
            layers.insert(
                layer_idx_pool,
                (
                    f'pool_pad_conv_{conv_block_id}',
                    nn.ReplicationPad1d(padding=(0, pool_size - 1)),
                ),
            )
        else:
            raise NotImplementedError()


        self.block = nn.Sequential(OrderedDict([*layers]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvStack(nn.Module):
    """Series of convolution blocks with optional dense connections."""

    def __init__(
        self,
        in_channels: int,
        conv_filters: List[int],
        conv_sizes: List[int],
        conv_strides: List[int],
        conv_padding: List[int],
        conv_padding_mode: str,
        pool_sizes: List[int],
        pool_strides: List[int],
        pool_padding: str,
        pool_before_bn_in_first_layer: bool,
    ):
        super().__init__()

        n_layers = len(conv_filters)

        conv_blocks = []
        for conv_block_id in range(n_layers):
            out_channels = conv_filters[conv_block_id]
            if conv_block_id == 0 and pool_before_bn_in_first_layer:
                pool_before_bn = True
            else:
                pool_before_bn = False

            conv_blocks.append(
                (
                    f'block_conv_{conv_block_id}',
                    ConvBlock(
                        conv_block_id=conv_block_id,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        conv_size=conv_sizes[conv_block_id],
                        conv_stride=conv_strides[conv_block_id],
                        conv_padding=conv_padding,
                        conv_padding_mode=conv_padding_mode,
                        pool_size=pool_sizes[conv_block_id],
                        pool_stride=pool_strides[conv_block_id],
                        pool_padding=pool_padding,
                        pool_before_bn=pool_before_bn,
                    ),
                ),
            )
            # set number of out_channels as new in_channels for next loop
            in_channels = out_channels

        self.stack = nn.Sequential(OrderedDict([*conv_blocks]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        return x


class ConvNet(Base):
    def __init__(self, output_units,
                 n_channels, sequence_length,
                 conv_filters, conv_sizes, conv_strides,
                 conv_padding, conv_padding_mode,
                 pool_sizes, pool_strides, pool_padding,
                 dense_units,
                 pool_before_bn_in_first_layer: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_func = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
        })

        if len(conv_filters) != len(conv_sizes):
            raise ValueError(
                f'len(conv_filters) != len(conv_sizes): '
                f'{len(conv_filters)} != {len(conv_sizes)}'
            )
        if len(conv_filters) != len(conv_strides):
            raise ValueError(
                f'len(conv_filters) != len(conv_strides): '
                f'{len(conv_filters)} != {len(conv_strides)}'
            )
        if len(conv_filters) != len(pool_sizes):
            raise ValueError(
                f'len(conv_filters) != len(pool_sizes): '
                f'{len(conv_filters)} != {len(pool_sizes)}'
            )
        if len(conv_filters) != len(pool_strides):
            raise ValueError(
                f'len(conv_filters) != len(pool_strides): '
                f'{len(conv_filters)} != {len(pool_strides)}'
            )

        self.conv_stack = ConvStack(
            in_channels=n_channels,
            conv_filters=conv_filters,
            conv_sizes=conv_sizes,
            conv_strides=conv_strides,
            conv_padding=conv_padding,
            conv_padding_mode=conv_padding_mode,
            pool_sizes=pool_sizes,
            pool_strides=pool_strides,
            pool_padding=pool_padding,
            pool_before_bn_in_first_layer=pool_before_bn_in_first_layer,
        )

        # create single instance mocking batch to get output shape of conv stack
        mock_input = torch.zeros((1, n_channels, sequence_length))
        mock_output = self.conv_stack(mock_input)
        fc_in_features = np.prod(mock_output.shape)

        # create layers for fully connected stack
        fc_layers = []

        # create each block with a dense, relu and batchnorm layer
        for fc_block_id in range(len(dense_units)):
            fc_out_features = dense_units[fc_block_id]
            fc_layers.extend([
                (
                    f'fc_{fc_block_id}',
                    nn.Linear(
                        in_features=fc_in_features,
                        out_features=fc_out_features,
                    ),
                ),
                (
                    f'act_fc_{fc_block_id}',
                    nn.ReLU(),
                ),
                (
                    f'bn_fc_{fc_block_id}',
                    nn.BatchNorm1d(
                        num_features=fc_out_features,
                        momentum=0.01,
                        eps=0.001,
                    ),
                ),
            ])
            # set number of out_features as new in_features for next loop
            fc_in_features = fc_out_features

        self.fc_stack = nn.Sequential(OrderedDict([*fc_layers]))

        self.classification_layer = nn.Linear(
            in_features=fc_in_features,
            out_features=output_units,
        )

        self.apply(init_weights)

    def forward(self, x, classification: bool = True):
        # apply stacks
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_stack(x)

        if classification:
            x = self.classification_layer(x)

        return x

    def configure_optimizers(self):
        learning_rate = 0.001
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
