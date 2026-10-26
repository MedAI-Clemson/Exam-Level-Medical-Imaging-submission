"""Pretraining configuration tests that need only the Python standard library."""

from collections import OrderedDict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretraining import configure_pretraining, normalize_classifier_options, normalize_model_state_dict, radimagenet_state_dict


class PretrainingConfigTests(unittest.TestCase):
    def config(self, source="imagenet", **extra):
        return {"mil": True, "pretraining": source, "encoder_kwargs": {"model_name": "resnet50"}, **extra}

    def test_each_source_has_an_explicit_configuration(self):
        for source in ("imagenet", "radimagenet", "instance"):
            with self.subTest(source=source):
                cfg = self.config(source, radimagenet_checkpoint="weights.pt", pretrained_model_dir="folds")
                configure_pretraining(cfg)
                self.assertEqual(cfg["pretraining"], source)
                self.assertFalse(cfg["freeze_enc"])

    def test_freezing_is_independent_of_source(self):
        for source in ("imagenet", "radimagenet", "instance"):
            cfg = self.config(source, freeze_enc=True, radimagenet_checkpoint="weights.pt", pretrained_model_dir="folds")
            configure_pretraining(cfg)
            self.assertTrue(cfg["freeze_enc"])

    def test_legacy_imagenet_config_is_translated(self):
        cfg = {"mil": True, "encoder_kwargs": {"model_name": "resnet50d", "pretrained": True}}
        configure_pretraining(cfg)
        self.assertEqual(cfg["pretraining"], "imagenet")
        self.assertNotIn("pretrained", cfg["encoder_kwargs"])

    def test_legacy_instance_config_keeps_clf_transfer(self):
        cfg = {"mil": True, "encoder_kwargs": {"model_name": "resnet50d", "pretrained": True},
               "pretrain_enc": True, "pretrain_fc": True, "pretrained_model_dir": "folds"}
        configure_pretraining(cfg)
        self.assertEqual(cfg["pretraining"], "instance")
        self.assertTrue(cfg["instance_load_clf"])
        self.assertNotIn("pretrain_enc", cfg)

    def test_old_classifier_options_become_canonical_clf_options(self):
        cfg = self.config("instance", pretrained_model_dir="folds", instance_load_head=True, freeze_fc=True)
        configure_pretraining(cfg)
        self.assertTrue(cfg["instance_load_clf"])
        self.assertTrue(cfg["freeze_clf"])
        self.assertNotIn("instance_load_head", cfg)
        self.assertNotIn("freeze_fc", cfg)

    def test_conflicting_classifier_aliases_fail(self):
        for old, new in (("instance_load_head", "instance_load_clf"), ("freeze_fc", "freeze_clf")):
            with self.assertRaisesRegex(ValueError, "Conflicting classifier settings"):
                configure_pretraining(self.config(**{old: True, new: False}))

    def test_normalized_old_options_allow_explicit_overrides(self):
        cfg = self.config(freeze_fc=True)
        normalize_classifier_options(cfg)
        cfg["freeze_clf"] = False
        configure_pretraining(cfg)
        self.assertFalse(cfg["freeze_clf"])

    def test_old_checkpoint_names_and_metadata_are_migrated(self):
        state = OrderedDict({"head.1.weight": "classifier", "cls": "token",
                             "encoder.head.weight": "encoder_internal", "query": "attention"})
        state._metadata = {"head": {"version": 1}, "encoder.head": {"version": 2}}
        weights = normalize_model_state_dict(state)
        self.assertEqual(weights["clf.1.weight"], "classifier")
        self.assertEqual(weights["bag_token"], "token")
        self.assertEqual(weights["encoder.head.weight"], "encoder_internal")
        self.assertEqual(weights["query"], "attention")
        self.assertEqual(weights._metadata["clf"], {"version": 1})
        self.assertIn("head.1.weight", state)  # The caller's checkpoint is not mutated.

    def test_mixed_checkpoint_names_do_not_overwrite_each_other(self):
        with self.assertRaisesRegex(ValueError, "Duplicate checkpoint parameter"):
            normalize_model_state_dict({"head.1.weight": "old", "clf.1.weight": "new"})

    def test_unknown_source_and_nonboolean_freezing_fail(self):
        for cfg in (self.config("unknown"), self.config(freeze_enc="false")):
            with self.assertRaises(ValueError):
                configure_pretraining(cfg)

    def test_radimagenet_requires_matching_architecture_and_weights(self):
        with self.assertRaisesRegex(ValueError, "radimagenet_checkpoint"):
            configure_pretraining(self.config("radimagenet"))
        cfg = self.config("radimagenet", radimagenet_checkpoint="weights.pt")
        cfg["encoder_kwargs"]["model_name"] = "resnet50d"
        with self.assertRaisesRegex(ValueError, "not resnet50d"):
            configure_pretraining(cfg)
        with self.assertRaisesRegex(ValueError, "TensorFlow/Keras"):
            configure_pretraining(self.config("radimagenet", radimagenet_checkpoint="weights.h5"))

    def test_instance_transfer_requires_mil_and_fold_directory(self):
        for cfg in (self.config("instance"), self.config("instance", mil=False, pretrained_model_dir="folds")):
            with self.assertRaisesRegex(ValueError, "Instance transfer requires"):
                configure_pretraining(cfg)

    def test_instance_training_can_start_from_either_external_source(self):
        for source in ("imagenet", "radimagenet"):
            cfg = self.config(source, mil=False, radimagenet_checkpoint="weights.pt")
            configure_pretraining(cfg)
            self.assertFalse(cfg["mil"])

    def test_clf_transfer_requires_compatible_classifiers(self):
        with self.assertRaisesRegex(ValueError, "requires pretraining: instance"):
            configure_pretraining(self.config(instance_load_clf=True))
        cfg = self.config("instance", pretrained_model_dir="folds", instance_load_clf=True,
                          net_kwargs={"pooling_method": "transformer"})
        with self.assertRaisesRegex(ValueError, "transformer classifier"):
            configure_pretraining(cfg)

    def test_official_radimagenet_parameter_names_are_mapped(self):
        source = {"module.backbone.0.weight": "conv", "module.backbone.1.running_mean": "bn",
                  "module.backbone.4.0.conv1.weight": "block", "module.backbone.7.2.bn3.bias": "last"}
        expected = {"conv1.weight": "conv", "bn1.running_mean": "bn",
                    "layer1.0.conv1.weight": "block", "layer4.2.bn3.bias": "last"}
        for wrapper in (source, {"state_dict": source}, {"model_state_dict": source}):
            self.assertEqual(radimagenet_state_dict(wrapper), expected)

    def test_canonical_names_are_retained_and_classifier_is_discarded(self):
        state = {"conv1.weight": "conv", "fc.weight": "classifier", "fc.bias": "bias"}
        self.assertEqual(radimagenet_state_dict(state), {"conv1.weight": "conv"})

    def test_bad_or_colliding_checkpoint_keys_fail(self):
        for state in ({}, {"backbone.99.weight": "bad"}, {"conv1.weight": 1, "backbone.0.weight": 2}):
            with self.assertRaises(ValueError):
                radimagenet_state_dict(state)


if __name__ == "__main__":
    unittest.main()
