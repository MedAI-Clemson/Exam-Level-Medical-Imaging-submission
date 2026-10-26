"""Image transforms operate on float32 images in [0, 1]."""

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ImageTransform:
    def __init__(self, size=224, train=False, rotation=10.0, mean=0.5, std=0.5):
        if size <= 0 or std <= 0 or rotation < 0:
            raise ValueError("size/std must be positive and rotation nonnegative")

        self.size, self.train, self.rotation = size, train, rotation
        self.mean, self.std = mean, std

    def __call__(self, image):
        # All images enter as float32 in [0, 1]; resizing preserves that representation.
        image = TF.resize(image, [self.size, self.size], antialias=True)

        # Draw a fresh rotation only during training. Border fill becomes zero after normalization.
        if self.train and self.rotation:
            angle = (torch.rand(()).item() * 2 - 1) * self.rotation
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR,
                              fill=self.mean)

        # Both training and validation use the same final input scale.
        return TF.normalize(image, [self.mean], [self.std])
