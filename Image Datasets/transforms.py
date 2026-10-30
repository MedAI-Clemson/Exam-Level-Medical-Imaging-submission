"""Resize and augment scaled grayscale images, then normalize for the encoder."""

import torch
from torchvision.transforms import RandAugment
from torchvision.transforms import functional as TF


class ImageTransform:
    def __init__(self, size=224, train=False, num_ops=2, magnitude=9, mean=0.5, std=0.5):
        if size <= 0 or std <= 0:
            raise ValueError("size/std must be positive")
        if type(num_ops) is not int or num_ops < 0:
            raise ValueError("num_ops must be a nonnegative integer")
        if type(magnitude) is not int or not 0 <= magnitude <= 30:
            raise ValueError("magnitude must be an integer from 0 to 30")

        self.size, self.train = size, train
        self.mean, self.std = mean, std
        # Match the original RandAugment defaults, including interpolation and fill.
        self.augment = RandAugment(num_ops=num_ops, magnitude=magnitude) if num_ops else None

    def __call__(self, image):
        # Decoding has already scaled PNG/JPEG or windowed DICOM pixels to [0, 1].
        image = TF.resize(image, [self.size, self.size], antialias=True)

        if self.train and self.augment is not None:
            # RandAugment tensor operations require uint8, including equalize/posterize.
            # Quantize only the scaled training image, never the raw DICOM intensities.
            image = (image.clamp(0, 1) * 255).round().to(torch.uint8)
            image = self.augment(image).to(torch.float32) / 255

        # Validation skips augmentation and retains the decoded floating-point precision.
        return TF.normalize(image, [self.mean], [self.std])
