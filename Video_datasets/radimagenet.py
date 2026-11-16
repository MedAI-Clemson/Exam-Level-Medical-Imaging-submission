import torch

# RadImageNet ResNet50 stores a torchvision-style resnet50 as nn.Sequential
# (backbone.0=conv1, 1=bn1, 4..7=layer1..layer4), so keys are remapped to timm's resnet50
_STAGE_MAP = {'0': 'conv1', '1': 'bn1', '4': 'layer1', '5': 'layer2', '6': 'layer3', '7': 'layer4'}

def load_radimagenet_resnet50(encoder, path):
    """Load RadImageNet ResNet50 weights (Lab-Rasool/RadImageNet) into a timm 'resnet50' encoder."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    remapped = {}
    for k, v in sd.items():
        _, idx, rest = k.split('.', 2)
        remapped[f'{_STAGE_MAP[idx]}.{rest}'] = v
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    # timm's resnet50 has no classifier when num_classes=0, so nothing should be missing
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    return encoder
