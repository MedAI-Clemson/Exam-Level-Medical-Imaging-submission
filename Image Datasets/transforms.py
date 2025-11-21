from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F



# ------ Transforms for Train Dataset --------
transform_augment = transforms.Compose([
                                transforms.Grayscale(),
                                transforms.Resize((224,224)),
                                transforms.RandAugment()
                                ])

# ------ Transforms for Validation Dataset --------
transform_original = transforms.Compose([
                                 transforms.Grayscale(),
                                 transforms.Resize((224,224)),
                                ])
