#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
ReDimNet wrapper for SASV baseline
ReDimNet: Reshape Dimensions Network for Speaker Recognition (Interspeech 2024)
Repository: https://github.com/IDRnD/redimnet
"""

import torch
import torch.nn as nn
from typing import Optional


class ReDimNetWrapper(nn.Module):
    """
    Wrapper for ReDimNet models loaded from torch.hub

    ReDimNet processes raw audio directly (16kHz) and outputs speaker embeddings.
    Unlike other models in this codebase, it does not require manual mel-spectrogram extraction.

    Args:
        model_name: ReDimNet variant ('b0', 'b1', 'b2', 'b3', 'b5', 'b6', 'S', 'M')
        train_type: Training type ('ptn' for pretraining, 'ft_lm' for Large-Margin fine-tuning, 'ft_mix' for mixed)
        dataset: Dataset used for pretraining ('vox2', 'vb2', 'vb2+vox2+cnc')
        num_out: Expected output embedding dimension (default 192)
        pretrained: Whether to load pretrained weights
    """
    def __init__(
        self,
        model_name: str = 'b2',
        train_type: str = 'ptn',
        dataset: str = 'vox2',
        num_out: int = 192,
        pretrained: bool = True,
        **kwargs
    ):
        super(ReDimNetWrapper, self).__init__()

        self.model_name = model_name
        self.train_type = train_type
        self.dataset = dataset
        self.num_out = num_out

        print(f"Loading ReDimNet model: {model_name}, train_type: {train_type}, dataset: {dataset}")

        # Load ReDimNet from torch.hub
        try:
            if pretrained:
                self.redimnet = torch.hub.load(
                    'IDRnD/ReDimNet',
                    'ReDimNet',
                    model_name=model_name,
                    train_type=train_type,
                    dataset=dataset,
                    trust_repo=True
                )
            else:
                # Load architecture without pretrained weights
                self.redimnet = torch.hub.load(
                    'IDRnD/ReDimNet',
                    'ReDimNet',
                    model_name=model_name,
                    train_type=None,  # No pretrained weights
                    dataset=None,
                    trust_repo=True
                )
            print(f"Successfully loaded ReDimNet-{model_name}")
        except Exception as e:
            print(f"Error loading ReDimNet from torch.hub: {e}")
            print("Attempting to load without trust_repo flag...")
            # Fallback without trust_repo
            self.redimnet = torch.hub.load(
                'IDRnD/ReDimNet',
                'ReDimNet',
                model_name=model_name,
                train_type=train_type if pretrained else None,
                dataset=dataset if pretrained else None
            )

        # Check if output dimension needs adjustment
        # ReDimNet outputs 192-dim embeddings by default
        # If num_out is different, add a linear projection layer
        if num_out != 192:
            print(f"Adding projection layer: 192 -> {num_out}")
            self.projection = nn.Sequential(
                nn.Linear(192, num_out),
                nn.BatchNorm1d(num_out)
            )
        else:
            self.projection = None

    def forward(self, x, aug=False):
        """
        Forward pass

        Args:
            x: Input raw audio tensor [batch_size, samples] at 16kHz
            aug: Augmentation flag (ignored for ReDimNet as it processes raw audio)

        Returns:
            embeddings: Speaker embeddings [batch_size, num_out]
        """
        # ReDimNet expects raw audio at 16kHz
        # Shape: [batch_size, samples]

        # Get embeddings from ReDimNet
        embeddings = self.redimnet(x)

        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


def MainModel(
    redimnet_model: str = 'b2',
    redimnet_train_type: str = 'ptn',
    redimnet_dataset: str = 'vox2',
    redimnet_pretrained: bool = True,
    num_out: int = 192,
    **kwargs
):
    """
    Factory function to create ReDimNet model

    Args:
        redimnet_model: Model variant ('b0', 'b1', 'b2', 'b3', 'b5', 'b6')
            - b0: 1.0M params, 0.43 GMACs
            - b1: 2.2M params, 0.54 GMACs
            - b2: 4.7M params, 0.90 GMACs (default)
            - b3: 3.0M params, 3.00 GMACs
            - b5: 9.2M params, 9.87 GMACs
            - b6: 15.0M params, 20.27 GMACs
        redimnet_train_type: Training type ('ptn', 'ft_lm', 'ft_mix')
        redimnet_dataset: Dataset ('vox2', 'vb2', 'vb2+vox2+cnc')
        redimnet_pretrained: Whether to load pretrained weights
        num_out: Output embedding dimension
        **kwargs: Additional arguments (for compatibility)

    Returns:
        ReDimNetWrapper model instance
    """
    model = ReDimNetWrapper(
        model_name=redimnet_model,
        train_type=redimnet_train_type,
        dataset=redimnet_dataset,
        num_out=num_out,
        pretrained=redimnet_pretrained,
        **kwargs
    )
    return model
