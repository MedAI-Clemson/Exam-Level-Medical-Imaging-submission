"""Map each source CSV to one manifest, then load images lazily."""

from pathlib import Path
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset

from imaging import load_image, ordered_slices, select_k_slices
from transforms import ImageTransform


def read_metadata(cfg):
    """Return patient_id, bag_id, label, source, and optional ijk columns."""
    task = cfg["task"]

    # Read IDs as strings so values such as "001" retain their leading zeros.
    data = pd.read_csv(cfg["metadata_csv"], dtype=str)

    # Each adapter identifies the patient, the prediction unit (bag), and its label.
    if task == "achilles":
        # Keep the original exclusions and encode unhealthy as the positive class.
        data = data.loc[pd.to_numeric(data["Drop"]) == 0].drop_duplicates("Image Name")
        data = data.loc[~data["Folder Name"].str.contains("bilateral", case=False, na=False)]
        healthy = data["Healthy?"].str.strip().str.lower().map(
            {"true": 0, "1": 0, "1.0": 0, "false": 1, "0": 1, "0.0": 1})
        patient, bag, source, labels = "Patient #", "Folder Name", "Given Name", healthy

    elif task == "xray":
        targets = pd.read_csv(cfg["targets_csv"], dtype=str)
        columns = ["patient_id"] + (["num_parts"] if "num_parts" in targets else [])
        targets = targets[columns].drop_duplicates()

        if "num_parts" in targets and "num_parts" in data:
            # Verify overlapping labels instead of relying on an implicit multi-column merge.
            check = data.merge(targets, on="patient_id", suffixes=("_image", "_target"),
                               validate="many_to_one")
            if not pd.to_numeric(check["num_parts_image"]).equals(
                    pd.to_numeric(check["num_parts_target"])):
                raise ValueError("X-ray image and target CSV labels disagree")
            data = data.drop(columns="num_parts")

        # Several images can share one patient's target, but none may lack a target.
        data = data.merge(targets, on="patient_id", validate="many_to_one", how="left",
                          indicator=True)
        if (data["_merge"] != "both").any():
            raise ValueError("Some X-ray images have no matching target patient")

        patient, bag, source = "patient_id", "patient_id", "path"
        labels = pd.to_numeric(data["num_parts"]).map({1: 0, 2: 1, 3: 2, 4: 2})

    elif task == "mri":
        # Combine a patient's selected series into one bag.
        patient, bag, source = "ProxID", "ProxID", "SeriesUID"
        labels = pd.to_numeric(data["ClinSig"])

    elif task == "ct":
        # Keep each series as a bag, while retaining patient IDs for fold isolation.
        patient, bag, source = "PatientID", "SeriesUID", "SeriesUID"
        labels = pd.to_numeric(data["Diagnosis"]).map({0: 0, 1: 1, 2: 1})
    else:
        raise ValueError(f"Unknown task: {task}")

    # From here onward, every modality uses the same metadata column names.
    result = pd.DataFrame({"patient_id": data[patient], "bag_id": data[bag],
                           "label": labels, "source": data[source]})
    if cfg.get("use_k", False):
        result["ijk"] = data["ijk"]

    # Filter before creating folds so excluded bags do not affect stratification.
    if cfg.get("view_filter"):
        view = {"xray": "view_group", "mri": "DCMSerDescr"}[task]
        before = set(result.bag_id)
        result = result.loc[data[view].isin(cfg["view_filter"])]
        excluded = len(before - set(result.bag_id))
        if excluded:
            warnings.warn(f"View filtering excluded {excluded} bags with no eligible images")

    # Reject incomplete or ambiguous records before any image is decoded.
    required = ["patient_id", "bag_id", "label", "source"]
    if result.empty or result[required].isna().any().any():
        raise ValueError("Metadata is empty or has missing IDs, labels, or paths")
    for key in ("patient_id", "bag_id", "source"):
        result[key] = result[key].str.strip()
        if result[key].eq("").any():
            raise ValueError(f"Empty metadata field: {key}")

    classes = set(range(2 if cfg["encoder_kwargs"]["num_classes"] == 1
                        else cfg["encoder_kwargs"]["num_classes"]))
    if set(result.label) != classes:
        raise ValueError(f"Metadata must contain exactly the expected classes {sorted(classes)}")
    result["label"] = result.label.astype(int)

    # A bag must have exactly one owner and one target label.
    for key in ("patient_id", "label"):
        if result.groupby("bag_id")[key].nunique().gt(1).any():
            raise ValueError(f"Conflicting {key} values within a bag; resolve the metadata")

    return result.drop_duplicates().sort_values(["bag_id", "source"]).reset_index(drop=True)


def build_manifest(cfg):
    # Expand metadata into one row per image path, without loading pixel data.
    metadata = read_metadata(cfg)
    records, cache = [], {}

    for row in metadata.itertuples(index=False):
        path = (Path(cfg["data_dir"]) / row.source).resolve()

        if cfg["task"] in ("ct", "mri"):
            # Repeated annotations may reference the same series; read its headers once.
            if path not in cache:
                cache[path] = ordered_slices(path, cfg.get("slice_order", "position"))
            paths = cache[path]
            if cfg.get("use_k", False):
                paths = select_k_slices(paths, row.ijk, cfg["k_index_base"])
        else:
            if not path.is_file():
                raise FileNotFoundError(path)
            paths = [path]

        for image_path in paths:
            records.append({"patient_id": row.patient_id, "bag_id": row.bag_id,
                            "label": row.label, "path": str(image_path)})

    # Overlapping k-neighborhoods can repeat a slice within a bag; keep it only once.
    manifest = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)
    if manifest.groupby("path")["bag_id"].nunique().gt(1).any():
        raise ValueError("The same image belongs to multiple bags")

    return manifest


class ImagingDataset(Dataset):
    def __init__(self, manifest, cfg, train=False):
        self.task = cfg["task"]
        self.preprocessing = cfg.get("preprocessing", {})
        self.transform = ImageTransform(cfg.get("res", 224), train=train,
                                        **cfg.get("transform_kwargs", {}))

        # Store paths, not images: MIL samples contain a bag; instance samples contain one image.
        if cfg["mil"]:
            self.samples = [(bag_id, group.patient_id.iloc[0], int(group.label.iloc[0]),
                             group.path.tolist())
                            for bag_id, group in manifest.groupby("bag_id", sort=True)]
        else:
            # Instance pretraining assigns every image the label of its enclosing bag.
            # The manifest already checked that all images within that bag agree.
            self.samples = [(row.bag_id, row.patient_id, int(row.label), [row.path])
                            for row in manifest.itertuples(index=False)]

        if not self.samples:
            raise ValueError("Dataset contains no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bag_id, patient_id, label, paths = self.samples[index]

        # Decode and augment on every access, so each training epoch sees fresh rotations.
        images = [self.transform(load_image(path, self.task, self.preprocessing)) for path in paths]

        # Stacking preserves the channel axis: [images_in_bag, channels, height, width].
        return {"bag_id": bag_id, "patient_id": patient_id, "label": label,
                "instance_path": paths[0] if len(paths) == 1 else "",
                "images": torch.stack(images)}


def collate_bags(samples):
    # Concatenate real images only. The model adds padding later, after encoding.
    # Lengths preserve bag boundaries; IDs remain aligned with labels and predictions.
    return {"images": torch.cat([sample["images"] for sample in samples]),
            "lengths": [len(sample["images"]) for sample in samples],
            "labels": torch.tensor([sample["label"] for sample in samples]),
            **{key: [sample[key] for sample in samples]
               for key in ("bag_id", "patient_id", "instance_path")}}
