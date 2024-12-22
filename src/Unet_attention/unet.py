"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""
from __future__ import annotations

import warnings
from tqdm import tqdm
from collections.abc import Sequence
from typing import (
    Dict,
    Tuple,
    List,
    Union
)
from omegaconf import OmegaConf
import torch
from torch import nn
import lightning as L
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers.factories import Act, Norm
from monai.utils import (
    look_up_option, 
    SkipMode
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.blocks import ResidualUnit, Convolution



__all__ = ["UNet", "Unet"]


class LightningSegmentationModel(L.LightningModule):
    def __init__(self, cfg: OmegaConf = None):
        super().__init__()
        # this would save the model as hyperparameter, not desired!
        self.cfg = cfg
        self.model = self.get_unet(cfg.unet_config)
        
        self.save_hyperparameters({
            'cfg': cfg,
        })
        
        self.lr = cfg.lr
        self.patience = cfg.patience
        self.cfg = cfg
        self.loss = DiceCELoss(
            softmax=False if cfg.binary_target else True,
            sigmoid=True if cfg.binary_target else False,
            to_onehot_y=False if cfg.binary_target else True,
        )
        self.dsc = DiceMetric(include_background=False, reduction="none")
        self.IoU = MeanIoU(include_background=False, reduction="none")

    def get_unet(self, unet_config):
        return UNet(
            spatial_dims=unet_config.spatial_dims,
            in_channels=unet_config.in_channels,
            out_channels=unet_config.out_channels,
            channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
            strides=[2] * (unet_config.depth - 1),
            num_res_units=4
        )
    
    def forward(self, inputs, with_emb=False):
        """
        Forward pass with optional embedding output.
        """
        if with_emb:
            # Example of extracting embeddings
            e1 = self.model(inputs)  # Feature map after certain layers
            e2 = e1  # Replace with the required intermediate feature map
            return e1, e2
        else:
            return self.model(inputs)  # Standard forward pass

    def test_model(self, test_loader, device):
        """
        Test the model on the given test_loader and return metrics.
        """
        self.eval()
        all_metrics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Adjust batch handling based on DataLoader format
                if isinstance(batch, tuple):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs, targets = batch["input"], batch["target"]
                else:
                    raise TypeError(f"Unsupported batch format: {type(batch)}")

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.forward(inputs)
                loss = self.loss(outputs, targets)

                # Compute additional metrics
                metrics = {
                    "loss": loss.item(),
                    # Add more metrics as needed
                }
                all_metrics.append(metrics)

                # Manual logging (if needed)
                print(f"Batch {batch_idx}: {metrics}")

        # Aggregate results
        avg_metrics = {
            key: sum(metric[key] for metric in all_metrics) / len(all_metrics)
            for key in all_metrics[0]
        }
        print(f"Test Metrics: {avg_metrics}")
        return avg_metrics


    def training_step(self, batch, batch_idx):
        input = batch['data']
        target = batch['target']
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0.5) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()

        self.log_dict({
            'train_loss': loss,
            'train_dsc': dsc,
        })
        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()

        self.log_dict({
            'val_loss': loss,
            'val_dsc': dsc,
        })
        return {
            'loss': loss,
        }
    
    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()


        self.log_dict({
            'test_loss': loss,
            'test_dsc': dsc,
        })
        return {
            'loss': loss,
            'dsc': dsc
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        outputs = self(input)
        return outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }
    

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class AttentionBlock(nn.Module):
    """
    Attention Block for UNet. Implements a simple additive attention mechanism.
    """
    def __init__(self, in_channels, gate_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        Forward pass for the AttentionBlock.

        Args:
            x: Tensor from the encoder (skip connection).
            g: Tensor from the decoder (gating signal).

        Returns:
            Attention-weighted skip connection.
        """
        g1 = self.W_g(g)  # Transform gating signal
        x1 = self.W_x(x)  # Transform skip connection
        psi = self.relu(g1 + x1)  # Add and apply ReLU
        psi = self.psi(psi)  # Generate attention weights
        return x * psi  # Apply attention weights to the skip connection



class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("The length of `channels` should be no less than 2.")
        if len(strides) != len(channels) - 1:
            raise ValueError("The length of `strides` should equal `len(channels) - 1`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # Define encoder, bottleneck, and decoder
        self.encoder = nn.ModuleList(
            [
                self._get_down_layer(
                    in_channels if i == 0 else channels[i],
                    channels[i + 1],
                    strides[i],
                    is_top=(i == 0),
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.bottleneck = self._get_bottom_layer(channels[-1], channels[-1])

        self.decoder = nn.ModuleList(
        [
            self._get_up_layer(
                in_channels=channels[i + 1],
                out_channels=channels[i],
                strides=strides[i],
                is_top=(i == len(channels) - 2),
            )
            for i in range(len(channels) - 2, -1, -1)
        ]
    )



        self.final_conv = nn.Conv2d(
            in_channels=16,  # Match the output channels from the last decoder layer
            out_channels=self.out_channels,  # Number of final output channels
            kernel_size=1  # Use 1x1x1 convolution for dimensionality reduction
        )


    def _get_down_layer(self, in_channels, out_channels, strides, is_top):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_bottom_layer(self, in_channels, out_channels):
        return ResidualUnit(
            self.dimensions,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network.
        Adjusts for concatenated skip connections.
        """
        # Adjust in_channels to account for concatenation
        adjusted_in_channels = in_channels * 2  # Concatenated input

        conv = Convolution(
            self.dimensions,
            in_channels=adjusted_in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv


    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Override the connection block to include attention mechanism.
        """
        # Extract the out_channels of the last layer in down_path
        if isinstance(down_path, nn.Sequential):
            down_out_channels = down_path[-1].out_channels
        else:
            down_out_channels = down_path.out_channels

        # Extract the in_channels of the first layer in up_path
        if isinstance(up_path, nn.Sequential):
            up_in_channels = up_path[0].in_channels
        else:
            up_in_channels = up_path.in_channels

        # Create the attention block
        attention_block = AttentionBlock(
            in_channels=down_out_channels,
            gate_channels=up_in_channels,
            inter_channels=down_out_channels // 2
        )

        # Combine down, attention, and up paths
        class AttentionSkipConnection(nn.Module):
            def __init__(self, attention_block, submodule):
                super().__init__()
                self.attention_block = attention_block
                self.submodule = submodule

            def forward(self, x, g):
                """
                Forward pass for the AttentionSkipConnection.
                
                Args:
                    x: Skip connection from the encoder.
                    g: Gating signal from the decoder.
                
                Returns:
                    Output of the submodule after applying the attention block.
                """
                attention_output = self.attention_block(x, g)
                return self.submodule(attention_output)

        return nn.Sequential(
            down_path,
            AttentionSkipConnection(attention_block, subblock),
            up_path
        )

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            # print(f"Encoder output shape: {x.shape}")

        # Bottleneck
        x = self.bottleneck(x)
        # print(f"Bottleneck output shape: {x.shape}")

        # Decoder path
        for i, (up, skip) in enumerate(zip(self.decoder, reversed(skip_connections))):
            # print(f"Decoder input before concatenation: {x.shape}, Skip connection shape: {skip.shape}")
            x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
            # print(f"Decoder input after concatenation: {x.shape}")
            x = up(x)  # Process the concatenated tensor
            # print(f"Decoder output shape (layer {i}): {x.shape}")

        # print(f"Decoder output before final convolution: {x.shape}")
        # Final layer
        x = self.final_conv(x)
        # print(f"Final output shape: {x.shape}")
        return x



Unet = UNet


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str | SkipMode = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value
        self.swivel = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        x = self.swivel(x)
        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")