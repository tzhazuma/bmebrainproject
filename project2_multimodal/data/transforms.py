"""
Data transforms and augmentations for 3D medical imaging.
"""

import torch
import random
import numpy as np


class RandomFlip3D:
    """Random flip along specified axes."""

    def __init__(self, flip_axes=(0, 1, 2), prob=0.5):
        self.flip_axes = flip_axes
        self.prob = prob

    def __call__(self, tensor):
        """
        Args:
            tensor: [C, D, H, W]
        """
        if random.random() < self.prob:
            axis = random.choice(self.flip_axes) + 1  # +1 to skip C dim
            tensor = torch.flip(tensor, dims=(axis,))
        return tensor


class RandomRotate3D:
    """Random 90-degree rotation in-plane."""

    def __init__(self, prob=0.3):
        self.prob = prob
        self.k_values = [0, 1, 2, 3]

    def __call__(self, tensor):
        if random.random() < self.prob:
            k = random.choice(self.k_values)
            if k > 0:
                tensor = torch.rot90(tensor, k, dims=(-2, -1))
        return tensor


class RandomGaussianNoise:
    """Add gaussian noise with random std."""

    def __init__(self, std_range=(0, 0.05), prob=0.3):
        self.std_range = std_range
        self.prob = prob

    def __call__(self, tensor):
        if random.random() < self.prob:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            tensor = tensor + noise
        return tensor


class RandomIntensityShift:
    """Random intensity shift and scaling."""

    def __init__(self, shift_range=(-0.1, 0.1), scale_range=(0.9, 1.1), prob=0.3):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, tensor):
        if random.random() < self.prob:
            shift = random.uniform(*self.shift_range)
            scale = random.uniform(*self.scale_range)
            tensor = tensor * scale + shift
        return tensor


class ComposeTransforms:
    """Compose multiple transforms to apply consistently across modalities."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor):
        for t in self.transforms:
            tensor = t(tensor)
        return tensor


def get_training_transforms():
    return ComposeTransforms([
        RandomFlip3D(flip_axes=(0, 1), prob=0.5),
        RandomRotate3D(prob=0.3),
        RandomGaussianNoise(std_range=(0, 0.03), prob=0.3),
        RandomIntensityShift(prob=0.2),
    ])


def get_validation_transforms():
    return ComposeTransforms([])
