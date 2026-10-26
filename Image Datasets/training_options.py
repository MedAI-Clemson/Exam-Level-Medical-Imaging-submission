"""Fold-model selection and output settings, independent of the ML runtime."""

import math


MODEL_SELECTIONS = ("last", "best_loss")


def configure_model_output(cfg):
    cfg.setdefault("model_selection", "last")
    cfg.setdefault("save_models", True)
    if cfg["model_selection"] not in MODEL_SELECTIONS:
        raise ValueError(f"model_selection must be one of {MODEL_SELECTIONS}")
    if type(cfg["save_models"]) is not bool:
        raise ValueError("save_models must be true or false")


class FoldSelection:
    """Track the selected epoch without retaining a second copy of model weights."""

    def __init__(self, method="last"):
        if method not in MODEL_SELECTIONS:
            raise ValueError(f"model_selection must be one of {MODEL_SELECTIONS}")
        self.method = method
        self.epoch = None
        self.loss = float("inf")

    def update(self, loss, epoch):
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")

        # Lowest-loss selection keeps the earlier epoch when losses tie.
        if self.method == "best_loss" and loss >= self.loss:
            return False
        self.epoch, self.loss = epoch, loss
        return True
