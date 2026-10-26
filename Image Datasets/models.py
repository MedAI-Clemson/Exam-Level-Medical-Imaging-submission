"""Instance classifiers and MIL pooling; every model returns logits."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from pretraining import load_radimagenet, normalize_model_state_dict


POOLING_METHODS = ("attn", "tanh_attn", "avg", "max", "transformer", "stan")


def pad_bags(features, lengths):
    # features: [total_real_images, feature_count]; lengths: images in each bag.
    if not lengths or min(lengths) <= 0 or sum(lengths) != len(features):
        raise ValueError("Bag lengths must be positive and sum to the image count")

    # Split the encoded images back into bags, then zero-pad to the longest bag.
    # padded: [bags, max_images, features]; valid: [bags, max_images].
    padded = pad_sequence(features.split(lengths), batch_first=True)
    counts = torch.tensor(lengths, device=features.device)
    valid = torch.arange(padded.shape[1], device=features.device)[None, :] < counts[:, None]

    return padded, valid


class Classifier(nn.Module):
    """A common encoder/classifier interface for instance pretraining and bag training."""

    def __init__(self, encoder, num_features, num_out=1, pooling=None,
                 num_heads=1, drop_rate=0.0, attn_hidden_size=64):
        super().__init__()
        if pooling is not None and pooling not in POOLING_METHODS:
            raise ValueError(f"Unknown pooling method: {pooling}")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")

        self.encoder = encoder
        self.pooling = pooling
        self.num_out = num_out
        self.num_heads = num_heads
        self.freeze_encoder = False
        clf_features = num_features

        # Each attention head learns weights from its own subset of image features.
        if pooling in ("attn", "tanh_attn"):
            if num_features % num_heads:
                raise ValueError("Encoder feature count must be divisible by num_heads")
            self.subspace = num_features // num_heads
            if pooling == "attn":
                self.query = nn.Parameter(torch.randn(num_heads, self.subspace))
            else:
                self.V = nn.Parameter(torch.randn(num_heads, attn_hidden_size, self.subspace))
                self.w = nn.Parameter(torch.randn(num_heads, attn_hidden_size))

        # The transformer uses a smaller feature space and a learned token for the bag.
        if pooling == "transformer":
            clf_features = 64
            if clf_features % num_heads:
                raise ValueError("Transformer width (64) must be divisible by num_heads")
            self.projection = nn.Linear(num_features, clf_features)
            layer = nn.TransformerEncoderLayer(clf_features, num_heads, 128,
                                               dropout=drop_rate, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.bag_token = nn.Parameter(torch.randn(1, 1, clf_features))

        # The classifier converts the image or pooled bag description into class logits.
        self.clf = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(clf_features, num_out))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Read old workflow checkpoint names while saving new checkpoints with clf.* keys.
        return super().load_state_dict(normalize_model_state_dict(state_dict), strict=strict, assign=assign)

    def set_encoder_frozen(self, frozen):
        # Freeze both gradient updates and training-only behavior inside the encoder.
        self.freeze_encoder = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # Also freeze BatchNorm buffers and encoder dropout.

        return self

    def forward(self, images, lengths):
        # A feature vector is a learned numerical description of one image.
        # The encoder maps [images, channels, height, width] to [images, features].
        features = self.encoder(images)
        if features.ndim != 2:
            raise ValueError("Encoder must return [number_of_images, number_of_features]")

        # Instance training classifies each image directly, without combining images.
        if self.pooling is None:
            return self.clf(features)

        # Standard MIL combines instance probabilities rather than feature vectors.
        if self.pooling == "stan":
            return self.standard_pool(self.clf(features), lengths)

        if self.pooling == "transformer":
            padded, valid = pad_bags(self.projection(features), lengths)

            # Prepend the bag token. It is always valid; padded image tokens are masked.
            tokens = torch.cat([self.bag_token.expand(len(lengths), -1, -1), padded], dim=1)
            valid = F.pad(valid, (1, 0), value=True)
            return self.clf(self.transformer(tokens, src_key_padding_mask=~valid)[:, 0])

        # Feature pooling combines real images into one feature vector per bag.
        padded, valid = pad_bags(features, lengths)

        if self.pooling == "avg":
            # Padded zeros add nothing, but the denominator must count only real images.
            pooled = padded.sum(1) / valid.sum(1, keepdim=True)

        elif self.pooling == "max":
            # Padding must not beat negative features when selecting a maximum.
            pooled = padded.masked_fill(~valid[..., None], -torch.inf).max(1).values

        else:
            # h: [bags, max_images, heads, features_per_head].
            h = padded.reshape(*padded.shape[:2], self.num_heads, self.subspace)
            if self.pooling == "attn":
                scores = (h * self.query).sum(-1) / math.sqrt(self.subspace)
            else:
                hidden = torch.einsum("blhd,hkd->blhk", h, self.V).tanh()
                scores = torch.einsum("blhk,hk->blh", hidden, self.w)

            # Softmax runs across images. Masking gives padding zero attention weight.
            weights = scores.masked_fill(~valid[..., None], -torch.inf).softmax(dim=1)
            pooled = (h * weights[..., None]).sum(1).flatten(1)

        return self.clf(pooled)

    def standard_pool(self, logits, lengths):
        # Noisy-OR in log space: log P(no positive instance) = sum log(1 - p_i).
        if self.num_out == 1:
            log_absent = F.logsigmoid(-logits)
        else:
            # Normalize across classes for each image, never across images.
            probability = logits.softmax(dim=-1)
            log_absent = torch.log1p(-probability.clamp(max=1 - torch.finfo(logits.dtype).eps))

        padded, _ = pad_bags(log_absent, lengths)  # Padding zero is neutral to the sum.
        absent = padded.sum(1)

        # log P(at least one positive) = log(1 - exp(log P(none))).
        # expm1 preserves precision when the probability of no positive is close to one.
        present = (-torch.expm1(absent)).clamp_min(torch.finfo(logits.dtype).tiny).log()

        if self.num_out == 1:
            return present - absent  # log odds, suitable for BCEWithLogitsLoss.

        # Softmax(log P(class present)) implements the original normalized noisy-OR.
        return present


def build_model(cfg):
    import timm

    # num_classes=0 asks timm for pooled image features without its classifier.
    encoder_options = dict(cfg["encoder_kwargs"])
    num_out = encoder_options.pop("num_classes")
    encoder_options.pop("pretrained", None)  # Weight choice comes from pretraining, not a second flag.
    source = cfg["pretraining"]
    encoder = timm.create_model(**encoder_options, num_classes=0, pretrained=source == "imagenet")

    # Instance mode loads its matching fold later; it must not download ImageNet weights first.
    if source == "radimagenet":
        load_radimagenet(encoder, cfg["radimagenet_checkpoint"])

    # Use the same encoder/classifier interface for pretraining and MIL weight transfer.
    options = dict(cfg.get("net_kwargs", {}))
    options.pop("debug", None)
    if options.pop("num_out", num_out) != num_out:
        raise ValueError("net_kwargs.num_out and encoder_kwargs.num_classes must agree")

    pooling = options.pop("pooling_method", "attn") if cfg["mil"] else None
    options.pop("pooling_method", None)

    return Classifier(encoder, encoder.num_features, num_out, pooling=pooling, **options)
