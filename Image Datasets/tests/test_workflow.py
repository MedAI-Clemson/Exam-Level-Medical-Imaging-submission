"""Synthetic regression tests; no clinical data, GPU, or downloaded weights."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import torch
from torch import nn

from configuration import load_config
from datasets import ImagingDataset, collate_bags, read_metadata
from imaging import dicom_pixels, ordered_slices, select_k_slices
from models import Classifier, POOLING_METHODS, pad_bags
from train import main, run_epoch
from utils import classification_metrics, load_pretraining, make_folds, save_checkpoint


class PoolingTests(unittest.TestCase):
    def test_unrelated_bags_do_not_change_eval_predictions(self):
        torch.manual_seed(10)
        first = torch.randn(2, 4) - 2
        other = torch.randn(7, 4)
        for method in POOLING_METHODS:
            for outputs in (1, 3):
                with self.subTest(method=method, outputs=outputs):
                    model = Classifier(nn.Identity(), 4, outputs, pooling=method).eval()
                    with torch.no_grad():
                        alone = model(first, [2])
                        together = model(torch.cat([first, other]), [2, 7])[:1]
                    torch.testing.assert_close(alone, together, atol=2e-6, rtol=2e-5)

    def test_average_and_max_exclude_padding(self):
        images = torch.tensor([[-4.0], [-2.0], [10.0], [11.0], [12.0]])
        for method, expected in (("avg", -3.0), ("max", -2.0)):
            model = Classifier(nn.Identity(), 1, pooling=method)
            model.clf = nn.Identity()
            self.assertEqual(model(images, [2, 3])[0].item(), expected)

    def test_standard_multiclass_matches_hand_calculation(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        model = Classifier(nn.Identity(), 3, 3, pooling="stan")
        model.clf = nn.Identity()
        presence = 1 - (1 - logits.softmax(-1)).prod(0)
        expected = presence / presence.sum()
        torch.testing.assert_close(model(logits, [2]).softmax(-1)[0], expected)

    def test_standard_binary_and_large_bag_are_finite(self):
        model = Classifier(nn.Identity(), 1, pooling="stan")
        model.clf = nn.Identity()
        logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
        torch.testing.assert_close(model(logits, [2]).sigmoid(), torch.tensor([[0.75]]))
        output = model(torch.full((1000, 1), 20.0), [1000])
        self.assertTrue(torch.isfinite(output).all())

    def test_all_pooling_methods_backpropagate(self):
        for method in POOLING_METHODS:
            with self.subTest(method=method):
                model = Classifier(nn.Identity(), 4, pooling=method)
                inputs = torch.randn(4, 4, requires_grad=True)
                nn.BCEWithLogitsLoss()(model(inputs, [1, 3]), torch.tensor([[0.0], [1.0]])).backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_invalid_bag_lengths_fail(self):
        for lengths in ([], [0, 2], [1]):
            with self.assertRaises(ValueError):
                pad_bags(torch.ones(2, 4), lengths)

    def test_freezing_encoder_preserves_batchnorm_statistics(self):
        encoder = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.8))
        model = Classifier(encoder, 4, pooling="avg")
        model.set_encoder_frozen(True)
        model.train()
        self.assertFalse(encoder.training)
        before = encoder[0].running_mean.clone()
        model(torch.ones(3, 4), [1, 2])
        torch.testing.assert_close(encoder[0].running_mean, before)


class ImagingTests(unittest.TestCase):
    @staticmethod
    def dicom(array):
        ds = pydicom.Dataset()
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.Rows, ds.Columns = array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = array.astype("<u2").tobytes()
        return ds

    def test_ct_rescale_preserves_values_above_255(self):
        ds = self.dicom(np.array([[0, 256], [512, 1024]]))
        ds.RescaleSlope, ds.RescaleIntercept = 2, -1024
        pixels = dicom_pixels(ds, "ct", {"ct_window": [-1024, 1024]})
        torch.testing.assert_close(pixels, torch.tensor([[[0.0, 0.25], [0.5, 1.0]]]))

    def test_mri_precision_inversion_and_constant_image(self):
        ds = self.dicom(np.array([[0, 256], [512, 1024]]))
        settings = {"mri_percentiles": [0, 100]}
        regular = dicom_pixels(ds, "mri", settings)
        ds.PhotometricInterpretation = "MONOCHROME1"
        torch.testing.assert_close(dicom_pixels(ds, "mri", settings), 1 - regular)
        constant = self.dicom(np.ones((2, 2)))
        self.assertTrue(torch.isfinite(dicom_pixels(constant, "mri", settings)).all())

    def test_slice_order_uses_projected_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [directory / name for name in ("a.dcm", "b.dcm", "c.dcm")]
            for path in paths:
                path.touch()
            headers = {path: SimpleNamespace(ImageOrientationPatient=[0, 1, 0, 0, 0, 1],
                                             ImagePositionPatient=[x, 0, 0])
                       for path, x in zip(paths, [2, 0, 1])}
            with patch("imaging.pydicom.dcmread", side_effect=lambda p, **kw: headers[p]):
                self.assertEqual(ordered_slices(directory), [paths[1], paths[2], paths[0]])

    def test_instance_number_fallback_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("a.dcm", "b.dcm")]
            for path in paths:
                path.touch()
            headers = {paths[0]: SimpleNamespace(InstanceNumber=2),
                       paths[1]: SimpleNamespace(InstanceNumber=1)}
            with patch("imaging.pydicom.dcmread", side_effect=lambda p, **kw: headers[p]):
                self.assertEqual(ordered_slices(temporary), paths[::-1])
                headers[paths[0]].InstanceNumber = 1
                with self.assertRaises(ValueError):
                    ordered_slices(temporary)

    def test_k_slices_include_zero_and_use_explicit_base(self):
        paths = ["slice0", "slice1", "slice2", "slice3"]
        self.assertEqual(select_k_slices(paths, "0 0 0", 0), paths[:2])
        self.assertEqual(select_k_slices(paths, "0 0 1", 1), paths[:2])
        self.assertEqual(select_k_slices(paths, "0 0 3", 0), paths[2:])
        for k, base in (("0 0 4", 0), ("0 0 0", 1), ("0 0 1", None)):
            with self.assertRaises(ValueError):
                select_k_slices(paths, k, base)


class DataAndTrainingTests(unittest.TestCase):
    @staticmethod
    def manifest():
        return pd.DataFrame([{"patient_id": f"p{label}_{patient}",
                              "bag_id": f"p{label}_{patient}_v{visit}", "label": label,
                              "path": f"{label}_{patient}_{visit}.png"}
                             for label in (0, 1) for patient in range(8) for visit in (0, 1)])

    def test_folds_isolate_patients_and_cover_every_bag_once(self):
        manifest = self.manifest()
        folds = make_folds(manifest, 2, 123)
        validation_bags = []
        for train, val in folds:
            self.assertFalse(set(train.patient_id) & set(val.patient_id))
            self.assertEqual(set(val.label), {0, 1})
            validation_bags.extend(val.bag_id)
        self.assertCountEqual(validation_bags, manifest.bag_id)

    def test_dataset_loads_fresh_images_on_each_access(self):
        manifest = self.manifest().iloc[:2].copy()
        manifest["bag_id"] = "bag"
        cfg = {"task": "xray", "mil": True, "res": 4, "transform_kwargs": {"rotation": 0}}
        with patch("datasets.load_image", return_value=torch.ones(1, 4, 4)) as reader:
            dataset = ImagingDataset(manifest, cfg, train=True)
            self.assertEqual(reader.call_count, 0)
            first, second = dataset[0], dataset[0]
            self.assertEqual(reader.call_count, 4)
            batch = collate_bags([first, second])
            self.assertEqual(batch["images"].shape, (4, 1, 4, 4))
            self.assertEqual(batch["lengths"], [2, 2])

    def test_achilles_labels_are_mapped_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.csv"
            data = pd.DataFrame({"Drop": [0, 0, 0], "Image Name": ["a", "b", "c"],
                                 "Folder Name": ["v1", "v2", "v3"], "Patient #": ["p1", "p1", "p2"],
                                 "Given Name": ["a.png", "b.png", "c.png"], "Healthy?": [True, True, False]})
            cfg = {"task": "achilles", "metadata_csv": str(path), "encoder_kwargs": {"num_classes": 1}}
            data.to_csv(path, index=False)
            result = read_metadata(cfg)
            self.assertEqual(result.label.tolist(), [0, 0, 1])
            self.assertEqual(result.patient_id.tolist(), ["p1", "p1", "p2"])
            data.loc[1, "Folder Name"] = "v1"
            data.loc[1, "Healthy?"] = False
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting label"):
                read_metadata(cfg)

    def test_single_sample_binary_and_multiclass_epochs(self):
        for outputs, label in ((1, 1), (3, 2)):
            model = Classifier(nn.Identity(), 4, outputs)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            batch = {"images": torch.ones(1, 4), "lengths": [1], "labels": torch.tensor([label]),
                     "patient_id": ["p"], "bag_id": ["b"], "instance_path": ["image.png"]}
            with self.subTest(outputs=outputs):
                metrics, predictions = run_epoch(model, [batch], torch.device("cpu"), optimizer)
                self.assertTrue(np.isfinite(metrics["loss"]))
                self.assertEqual(len(predictions), 1)
                self.assertTrue(np.isnan(metrics["roc_auc"]))

    def test_xray_merge_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            images, targets = Path(temporary) / "images.csv", Path(temporary) / "targets.csv"
            pd.DataFrame({"patient_id": ["001", "002", "003"], "path": ["a", "b", "c"],
                          "num_parts": [1, 2, 4], "view_group": ["AP_LIKE"] * 3}).to_csv(images, index=False)
            data = pd.DataFrame({"patient_id": ["001", "002", "003"], "num_parts": [1, 2, 4]})
            data.to_csv(targets, index=False)
            cfg = {"task": "xray", "metadata_csv": str(images), "targets_csv": str(targets),
                   "encoder_kwargs": {"num_classes": 3}, "view_filter": ["AP_LIKE"]}
            result = read_metadata(cfg)
            self.assertEqual(result.label.tolist(), [0, 1, 2])
            self.assertEqual(result.patient_id.tolist(), ["001", "002", "003"])
            data.loc[0, "num_parts"] = 2
            data.to_csv(targets, index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                read_metadata(cfg)

    def test_cli_network_overrides_are_validated(self):
        path = Path(__file__).resolve().parents[1] / "config" / "achilles_config.yaml"
        with self.assertRaisesRegex(ValueError, "transformer classifier"):
            load_config(path, net_overrides={"pooling_method": "transformer"})
        with self.assertRaisesRegex(ValueError, "num_heads"):
            load_config(path, net_overrides={"num_heads": 0})

    def test_saved_checkpoint_is_independent_of_later_weight_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.ckpt"
            model = nn.Linear(1, 1)
            expected = model.weight.detach().clone()
            save_checkpoint(path, model, {"epoch": 1, "loss": 0.2})
            with torch.no_grad():
                model.weight.fill_(99)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["epoch"], 1)
            torch.testing.assert_close(saved["model_state_dict"]["weight"], expected)

    def test_pretraining_rejects_patient_leakage(self):
        model = Classifier(nn.Linear(4, 4), 4)
        cfg = {"task": "ct", "encoder_kwargs": {"model_name": "test"}, "pretraining": "instance"}
        checkpoint = {"format_version": 2, "task": "ct", "mil": False,
                      "encoder_name": "test", "num_out": 1,
                      "train_patients": ["validation_patient"], "val_patients": ["training_patient"]}
        with self.assertRaisesRegex(ValueError, "train_patients"):
            load_pretraining(model, checkpoint, cfg, ["training_patient"], ["validation_patient"])

    def test_all_supplied_configs_load_including_pretraining(self):
        for path in (Path(__file__).resolve().parents[1] / "config").glob("*.yaml"):
            with self.subTest(config=path.name):
                cfg = load_config(path)
                self.assertTrue(Path(cfg["data_dir"]).is_absolute())
                self.assertGreater(cfg["num_epochs"], 0)

    def test_classifier_cli_name_and_legacy_alias_share_one_setting(self):
        import io
        from contextlib import redirect_stdout
        from train import cli

        for option, expected in (("--instance-load-clf", True), ("--no-instance-load-clf", False),
                                 ("--instance-load-head", True), ("--no-instance-load-head", False)):
            with self.subTest(option=option), patch("sys.argv", ["train.py", "--config", "unused.yaml", option, "--check-config"]):
                with patch("train.load_config", return_value={}) as load, redirect_stdout(io.StringIO()):
                    cli()
                self.assertEqual(load.call_args.args[2]["instance_load_clf"], expected)

    def test_selection_and_saving_are_independent(self):
        def tiny_model(cfg):
            model = Classifier(nn.Linear(1, 1), 1, pooling="avg")
            model.test_epoch = 0
            return model

        def controlled_epoch(model, loader, device, optimizer=None):
            if optimizer is not None:
                model.test_epoch += 1
                with torch.no_grad():
                    next(model.parameters()).fill_(model.test_epoch)
                optimizer.step()
            loss = (0.3, 0.1, 0.8)[model.test_epoch - 1]
            return {"loss": loss}, pd.DataFrame({"probability_1": [model.test_epoch / 10]})

        with tempfile.TemporaryDirectory() as temporary:
            for method in ("last", "best_loss"):
                for save in (False, True):
                    with self.subTest(method=method, save=save):
                        output = Path(temporary) / f"{method}_{save}"
                        cfg = {"task": "achilles", "mil": True, "pretraining": "imagenet", "seed": 123,
                               "device": "cpu", "folds": 2, "num_epochs": 3, "res": 4,
                               "save_dir": str(output), "model_selection": method, "save_models": save,
                               "encoder_kwargs": {"model_name": "synthetic", "num_classes": 1},
                               "loader_kwargs": {"batch_size": 2, "num_workers": 0},
                               "optim_kwargs": {"lr": 0.001}, "sched_kwargs": {"step_size": 1}}
                        with patch("train.build_manifest", return_value=self.manifest()), \
                             patch("train.build_model", side_effect=tiny_model), \
                             patch("train.run_epoch", side_effect=controlled_epoch), \
                             patch("train.save_checkpoint", wraps=save_checkpoint) as writer:
                            main(cfg)
                        expected_epoch = 3 if method == "last" else 2
                        expected_saves = (2 if method == "last" else 4) if save else 0
                        self.assertEqual(writer.call_count, expected_saves)
                        summary = pd.read_csv(output / "summary.csv")
                        predictions = pd.read_csv(output / "predictions.csv")
                        self.assertEqual(summary.selected_epoch.tolist(), [expected_epoch] * 2)
                        self.assertEqual(predictions.epoch.tolist(), [expected_epoch] * 2)
                        np.testing.assert_allclose(predictions.probability_1, expected_epoch / 10)
                        for fold in (1, 2):
                            path = output / f"model_fold{fold}.ckpt"
                            self.assertEqual(path.exists(), save)
                            if save:
                                saved = torch.load(path, weights_only=True)
                                self.assertEqual(saved["epoch"], expected_epoch)
                                self.assertEqual(saved["model_state_dict"]["encoder.weight"].item(), expected_epoch)

    def test_two_fold_workflow_exports_selected_epoch_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {"task": "achilles", "mil": True, "pretraining": "imagenet", "seed": 123, "device": "cpu", "folds": 2,
                   "num_epochs": 2, "model_selection": "best_loss", "save_models": True,
                   "save_dir": str(Path(temporary) / "run"), "res": 4,
                   "transform_kwargs": {"rotation": 0}, "loader_kwargs": {"batch_size": 7, "num_workers": 0},
                   "encoder_kwargs": {"model_name": "synthetic", "num_classes": 1},
                   "optim_kwargs": {"lr": 0.001}, "sched_kwargs": {"step_size": 1, "gamma": 0.9}}
            def tiny_model(config):
                return Classifier(nn.Sequential(nn.Flatten(), nn.Linear(16, 4)), 4, pooling="avg")
            with patch("train.build_manifest", return_value=self.manifest()), \
                 patch("train.build_model", side_effect=tiny_model), \
                 patch("datasets.load_image", side_effect=lambda path, *args: torch.full((1, 4, 4), float(path[0]))):
                main(cfg)
            output = Path(cfg["save_dir"])
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(predictions), len(self.manifest()))
            for fold in (1, 2):
                checkpoint = torch.load(output / f"model_fold{fold}.ckpt", weights_only=True)
                history = pd.read_csv(output / f"metrics_fold{fold}.csv")
                best_epoch = int(history.loc[history.val_loss.idxmin(), "epoch"])
                self.assertEqual(checkpoint["epoch"], best_epoch)
                self.assertEqual(set(predictions.loc[predictions.fold == fold, "epoch"]), {best_epoch})


if __name__ == "__main__":
    unittest.main()
