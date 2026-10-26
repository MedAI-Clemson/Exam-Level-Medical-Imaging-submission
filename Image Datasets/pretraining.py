"""Encoder initialization choices and RadImageNet checkpoint compatibility."""

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path


PRETRAINING_OPTIONS = ("imagenet", "radimagenet", "instance")


def normalize_classifier_options(cfg):
    # Keep historical classifier option names readable, but save only the canonical names.
    for old, new in (("instance_load_head", "instance_load_clf"), ("freeze_fc", "freeze_clf")):
        if old in cfg:
            value = cfg.pop(old)
            if new in cfg and cfg[new] != value:
                raise ValueError(f"Conflicting classifier settings: {old} and {new}")
            cfg[new] = value


def configure_pretraining(cfg):
    normalize_classifier_options(cfg)

    # Translate older configs once. The resolved config uses one explicit source.
    encoder = cfg["encoder_kwargs"]
    legacy_encoder = cfg.pop("pretrain_enc", False)
    legacy_clf = cfg.pop("pretrain_fc", False)
    legacy_imagenet = encoder.pop("pretrained", True)
    if "pretraining" not in cfg:
        if legacy_clf and not legacy_encoder:
            raise ValueError("Classifier-only legacy transfer is ambiguous; select pretraining: instance")
        if not legacy_encoder and not legacy_imagenet:
            raise ValueError("Choose pretraining: imagenet, radimagenet, or instance")
        cfg["pretraining"] = "instance" if legacy_encoder else "imagenet"

    source = cfg["pretraining"]
    if source not in PRETRAINING_OPTIONS:
        raise ValueError(f"pretraining must be one of {PRETRAINING_OPTIONS}")

    # Instance weights always initialize the encoder. Copying the classifier is optional.
    cfg.setdefault("instance_load_clf", bool(legacy_clf) if source == "instance" else False)
    cfg.setdefault("freeze_enc", False)
    cfg.setdefault("freeze_clf", False)
    for key in ("instance_load_clf", "freeze_enc", "freeze_clf"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    if cfg["instance_load_clf"] and source != "instance":
        raise ValueError("instance_load_clf requires pretraining: instance")

    if source == "instance":
        if not cfg["mil"] or not cfg.get("pretrained_model_dir"):
            raise ValueError("Instance transfer requires mil: true and pretrained_model_dir")
        if cfg["instance_load_clf"] and cfg.get("net_kwargs", {}).get("pooling_method") == "transformer":
            raise ValueError("The transformer classifier cannot load an instance classifier")

    if source == "radimagenet":
        # The official checkpoint uses a standard ResNet-50 stem, not ResNet-50d's deep stem.
        if encoder["model_name"] != "resnet50":
            raise ValueError("RadImageNet requires encoder_kwargs.model_name: resnet50 (not resnet50d)")
        if not cfg.get("radimagenet_checkpoint"):
            raise ValueError("Set radimagenet_checkpoint to the official PyTorch ResNet-50 weights")
        if Path(cfg["radimagenet_checkpoint"]).suffix.lower() in (".h5", ".hdf5", ".keras"):
            raise ValueError("RadImageNet requires PyTorch weights, not a TensorFlow/Keras checkpoint")


def normalize_model_state_dict(state):
    # Older workflow checkpoints called the classifier 'head' and the bag token 'cls'.
    # Rename only our top-level keys, leaving encoder internals and attention heads alone.
    def rename(key):
        if key == "cls":
            return "bag_token"
        if key == "head" or key.startswith("head."):
            return "clf" + key[len("head"):]
        return key

    weights = OrderedDict()
    for key, value in state.items():
        canonical = rename(key)
        if canonical in weights:
            raise ValueError(f"Duplicate checkpoint parameter after renaming: {canonical}")
        weights[canonical] = value

    # PyTorch state dictionaries can carry per-module version metadata.
    if hasattr(state, "_metadata"):
        weights._metadata = {rename(key): value for key, value in state._metadata.items()}

    return weights


def radimagenet_state_dict(checkpoint):
    # Accept a raw state dict or a conventional checkpoint wrapper.
    if not isinstance(checkpoint, Mapping):
        raise ValueError("RadImageNet checkpoint must contain a state dictionary")
    for wrapper in ("state_dict", "model_state_dict"):
        if wrapper in checkpoint:
            checkpoint = checkpoint[wrapper]
            break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("RadImageNet state dictionary is empty or invalid")

    # Official Backbone wraps torchvision's modules in Sequential. Map its named
    # indices to timm's equivalent ResNet modules; never rely on dictionary order.
    modules = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    weights = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint parameter names must be strings")
        key = key.removeprefix("module.")
        if key.startswith("backbone."):
            index, separator, suffix = key[len("backbone."):].partition(".")
            if index not in modules or not separator:
                raise ValueError(f"Unrecognized RadImageNet backbone key: {key}")
            key = modules[index] + "." + suffix

        # These fc names belong to external torchvision checkpoints, not our clf module.
        # Classification labels differ across tasks; only the encoder is transferred.
        if key in ("fc.weight", "fc.bias"):
            continue
        if key in weights:
            raise ValueError(f"Duplicate checkpoint parameter after mapping: {key}")
        weights[key] = value

    return weights


def load_radimagenet(encoder, checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = radimagenet_state_dict(checkpoint)
    if not all(torch.is_tensor(value) for value in weights.values()):
        raise ValueError("RadImageNet state dictionary must contain tensors only")

    # Our images have one channel. Summing the RGB kernels is equivalent to
    # repeating the normalized grayscale image across the three original channels.
    first_conv = weights.get("conv1.weight")
    if first_conv is not None and first_conv.ndim == 4:
        if first_conv.shape[1] == 3 and encoder.conv1.in_channels == 1:
            weights["conv1.weight"] = first_conv.sum(dim=1, keepdim=True)

    # Missing layers or architecture mismatches must fail, not leave random weights behind.
    try:
        encoder.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ValueError("RadImageNet weights do not match standard ResNet-50; check the checkpoint format") from error
