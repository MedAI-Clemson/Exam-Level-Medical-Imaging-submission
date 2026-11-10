"""Method-specific image transforms and frozen feature extraction."""

import json
from pathlib import Path


class ImageReader:
    def __init__(self, cfg, dataset):
        self.cfg, self.dataset = cfg, dataset
        self.mnist = None
        if dataset['task'] == 'mnist':
            from torchvision.datasets import MNIST
            self.mnist = {split: MNIST(cfg['mnist_data_dir'], train=split == 'train', download=False)
                          for split in ('train', 'test')}

    def read(self, index, method, training=False):
        import torch
        from torchvision.transforms import functional as TF
        image = self.dataset['images'][index]
        if self.mnist is not None:
            tensor = TF.to_tensor(self.mnist[image['mnist_split']][image['index']][0])
        else:
            from imaging import load_image
            tensor = load_image(image['path'], self.dataset['task'], self.dataset['preprocessing'])
        if method == 'promil' and self.mnist is not None:
            return TF.normalize(TF.resize(tensor, [28, 28], antialias=True), [.1307], [.3081])
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if method == 'promil':
            # Adapt the paper's ultrasound preprocessing to medical modalities.
            size = 200 if training else 172
            tensor = TF.resize(tensor, [size, size], antialias=True)
            if training:
                tensor = TF.center_crop(tensor, [172, 172])
        else:
            tensor = TF.resize(tensor, [224, 224], antialias=True)
        return TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])


def extract_features(cfg, root, dataset_name):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import resnet50, ResNet50_Weights
    from design import digest, write_json

    if not torch.cuda.is_available():
        raise RuntimeError('Frozen feature extraction requires a GPU job')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    dataset = manifest['datasets'][dataset_name]
    reader = ImageReader(cfg, dataset)

    class Images(Dataset):
        def __len__(self):
            return len(dataset['images'])

        def __getitem__(self, index):
            return reader.read(index, 'features')

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Original CLAM/DTFD feature backbone ends at layer3 (1024 channels), not layer4.
    encoder = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                            backbone.layer1, backbone.layer2, backbone.layer3,
                            nn.AdaptiveAvgPool2d(1), nn.Flatten()).cuda().eval()
    loader = DataLoader(Images(), batch_size=cfg['feature_batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'], pin_memory=True)
    directory = root / 'features' / dataset_name
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'features.tmp.npy'
    matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(dataset['images']), 1024))
    offset = 0
    with torch.inference_mode():
        for images in loader:
            features = encoder(images.cuda(non_blocking=True)).cpu().numpy()
            if not np.isfinite(features).all():
                raise ValueError('Non-finite frozen features')
            matrix[offset:offset+len(features)] = features
            offset += len(features)
    if offset != len(dataset['images']):
        raise ValueError('Incomplete feature extraction')
    matrix.flush()
    del matrix
    temporary.replace(directory / 'features.npy')
    write_json(directory / 'complete.json', dict(images=offset, dimensions=1024,
                                                manifest_sha256=digest(root / 'manifest.json'),
                                                encoder='torchvision_resnet50_layer3_IMAGENET1K_V1'))


class ProMILBags:
    """Load a whole bag in a worker; keep one bag per optimizer update."""
    def __init__(self, reader, bags, training):
        self.reader, self.bags, self.training = reader, bags, training

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        import torch
        images = torch.stack([self.reader.read(i, 'promil', self.training)
                              for i in self.bags[index]['images']])
        return index, images
