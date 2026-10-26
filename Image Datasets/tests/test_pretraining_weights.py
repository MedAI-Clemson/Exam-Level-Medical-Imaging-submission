"""Synthetic weight-transfer tests; no pretrained downloads or clinical images."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch import nn

from datasets import ImagingDataset
from models import Classifier, build_model
from pretraining import load_radimagenet
from utils import load_pretraining


class PretrainingWeightTests(unittest.TestCase):
    def test_only_imagenet_requests_timm_pretrained_weights(self):
        for source in ("imagenet", "radimagenet", "instance"):
            cfg = {"mil": True, "pretraining": source, "radimagenet_checkpoint": "weights.pt",
                   "encoder_kwargs": {"model_name": "resnet50", "num_classes": 1, "in_chans": 1}}
            encoder = nn.Linear(4, 4)
            encoder.num_features = 4
            with patch.dict(sys.modules, {"timm": SimpleNamespace(create_model=lambda **kwargs: encoder)}):
                with patch("timm.create_model", return_value=encoder) as create, patch("models.load_radimagenet") as load:
                    model = build_model(cfg)
                    self.assertEqual(create.call_args.kwargs["pretrained"], source == "imagenet")
                    self.assertEqual(create.call_args.kwargs["num_classes"], 0)
                    self.assertEqual(load.call_count, int(source == "radimagenet"))
                    self.assertIs(model.encoder, encoder)

    def test_official_resnet50_layout_loads_strictly_with_grayscale_conversion(self):
        import timm
        from torchvision.models import resnet50

        # Match the official Backbone wrapper, using random weights rather than a download.
        reference = resnet50(weights=None)
        state = {"backbone." + key: value for key, value in nn.Sequential(*list(reference.children())[:9]).state_dict().items()}
        encoder = timm.create_model("resnet50", pretrained=False, num_classes=0, in_chans=1)
        with patch("torch.load", return_value=state):
            load_radimagenet(encoder, "synthetic.pt")
        torch.testing.assert_close(encoder.conv1.weight, reference.conv1.weight.sum(1, keepdim=True))
        torch.testing.assert_close(encoder.layer4[2].bn3.running_var, reference.layer4[2].bn3.running_var)

    def test_incomplete_radimagenet_weights_are_rejected(self):
        encoder = nn.Module()
        encoder.conv1 = nn.Conv2d(1, 2, 3)
        with patch("torch.load", return_value={"conv1.weight": torch.ones(2, 3, 3, 3)}):
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_radimagenet(encoder, "missing_bias.pt")

    def test_instance_encoder_and_optional_clf_transfer(self):
        for load_clf in (False, True):
            source = Classifier(nn.Linear(4, 4), 4)
            target = Classifier(nn.Linear(4, 4), 4)
            with torch.no_grad():
                for parameter in source.parameters():
                    parameter.fill_(0.25)
                target.clf[1].weight.zero_()
            old_clf = target.clf[1].weight.detach().clone()
            checkpoint = {"format_version": 2, "task": "ct", "mil": False, "encoder_name": "test",
                          "num_out": 1, "train_patients": ["train"], "val_patients": ["val"],
                          "instance_label_source": "bag", "model_state_dict": source.state_dict()}
            cfg = {"task": "ct", "pretraining": "instance", "instance_load_clf": load_clf,
                   "encoder_kwargs": {"model_name": "test"}}
            load_pretraining(target, checkpoint, cfg, ["train"], ["val"])
            torch.testing.assert_close(target.encoder.weight, source.encoder.weight)
            torch.testing.assert_close(target.clf[1].weight, source.clf[1].weight if load_clf else old_clf)

    def test_instance_samples_inherit_bag_labels(self):
        manifest = pd.DataFrame({"patient_id": ["p1", "p1", "p2", "p2"],
                                 "bag_id": ["b1", "b1", "b2", "b2"], "label": [0, 0, 1, 1],
                                 "path": ["a.png", "b.png", "c.png", "d.png"]})
        cfg = {"task": "ct", "mil": False}
        dataset = ImagingDataset(manifest, cfg, train=True)
        self.assertEqual([sample[2] for sample in dataset.samples], [0, 0, 1, 1])
        self.assertTrue(all(len(sample[3]) == 1 for sample in dataset.samples))

    def test_old_full_checkpoint_loads_and_saves_canonical_names(self):
        source = Classifier(nn.Linear(4, 4), 4, pooling="transformer")
        legacy = {("head." + key[4:] if key.startswith("clf.") else "cls" if key == "bag_token" else key): value
                  for key, value in source.state_dict().items()}
        target = Classifier(nn.Linear(4, 4), 4, pooling="transformer")
        target.load_state_dict(legacy)
        torch.testing.assert_close(target.clf[1].weight, source.clf[1].weight)
        torch.testing.assert_close(target.bag_token, source.bag_token)
        self.assertTrue(any(key.startswith("clf.") for key in target.state_dict()))
        self.assertFalse(any(key.startswith("head.") for key in target.state_dict()))

    def test_instance_transfer_accepts_old_classifier_parameter_names(self):
        source = Classifier(nn.Linear(4, 4), 4)
        target = Classifier(nn.Linear(4, 4), 4)
        legacy = {"head." + key[4:] if key.startswith("clf.") else key: value
                  for key, value in source.state_dict().items()}
        checkpoint = {"format_version": 2, "task": "ct", "mil": False, "encoder_name": "test",
                      "num_out": 1, "train_patients": ["train"], "val_patients": ["val"],
                      "model_state_dict": legacy}
        cfg = {"task": "ct", "encoder_kwargs": {"model_name": "test"}, "instance_load_clf": True}
        load_pretraining(target, checkpoint, cfg, ["train"], ["val"])
        torch.testing.assert_close(target.clf[1].weight, source.clf[1].weight)

    def test_freezing_preserves_encoder_weights_buffers_and_dropout(self):
        encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.8))
        model = Classifier(encoder, 4, pooling="avg")
        model.set_encoder_frozen(True)
        model.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

        before = {key: value.clone() for key, value in encoder.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        loss = nn.BCEWithLogitsLoss()(model(torch.randn(4, 4), [2, 2]), torch.tensor([[0.0], [1.0]]))
        loss.backward()
        optimizer.step()
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[key])

        model.set_encoder_frozen(False)
        self.assertTrue(encoder.training)
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
