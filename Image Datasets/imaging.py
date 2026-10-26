"""DICOM ordering, explicit slice selection, and intensity-preserving decoding."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut
import torch
from torchvision.io import ImageReadMode, read_image


def ordered_slices(directory, order="position"):
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() == ".dcm")
    if not paths:
        raise ValueError(f"No DICOM slices in {directory}")

    # Ordering needs only the headers; pixel decoding happens when a sample is loaded.
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    if any(int(getattr(ds, "NumberOfFrames", 1)) != 1 for ds in headers):
        raise ValueError(f"Expected single-frame DICOM slices in {directory}")
    series = {str(ds.SeriesInstanceUID) for ds in headers if hasattr(ds, "SeriesInstanceUID")}
    if len(series) > 1:
        raise ValueError(f"Multiple DICOM series in {directory}")

    # Project positions onto the slice normal so oblique scans are ordered correctly too.
    geometry = all(hasattr(ds, "ImagePositionPatient") and
                   hasattr(ds, "ImageOrientationPatient") for ds in headers)
    if order == "position" and geometry:
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if not np.isclose(np.linalg.norm(normal), 1, atol=1e-3):
            raise ValueError(f"Invalid DICOM orientation in {directory}")
        if any(not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4)
               for ds in headers):
            raise ValueError(f"Inconsistent slice orientations in {directory}")
        keys = [float(np.dot(ds.ImagePositionPatient, normal)) for ds in headers]

    elif order in ("position", "instance"):
        # InstanceNumber is the fallback when position/orientation metadata is absent.
        if not all(hasattr(ds, "InstanceNumber") for ds in headers):
            raise ValueError(f"Cannot determine slice order in {directory}")
        keys = [int(ds.InstanceNumber) for ds in headers]
    else:
        raise ValueError("slice_order must be 'position' or 'instance'")

    if len(set(keys)) != len(keys):
        raise ValueError(f"Ambiguous duplicate slice positions/numbers in {directory}")

    return [p for _, p in sorted(zip(keys, paths), key=lambda pair: pair[0])]


def select_k_slices(paths, ijk, index_base):
    """k is an index in the configured DICOM order, not a filename/InstanceNumber."""
    if index_base not in (0, 1):
        raise ValueError("k_index_base must explicitly be 0 or 1 when use_k is enabled")

    parts = str(ijk).split()
    if len(parts) != 3:
        raise ValueError(f"Expected three integer ijk coordinates, got {ijk!r}")

    # Convert the annotation's index to Python's zero-based indexing.
    k = int(parts[2]) - index_base
    if not 0 <= k < len(paths):
        raise ValueError(f"k={k} is outside a series with {len(paths)} slices")

    # Slice bounds retain the first/last image when a neighbor lies outside the series.
    return paths[max(0, k - 1):min(len(paths), k + 2)]


def dicom_pixels(ds, modality, preprocessing):
    # Apply the DICOM intensity mapping before scaling; uint8 would discard precision.
    pixels = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("Expected a finite, single-frame grayscale DICOM image")

    # CT retains a fixed intensity reference; MRI is scaled using each slice's percentiles.
    if modality == "ct":
        lower, upper = preprocessing["ct_window"]
    else:
        lower, upper = np.percentile(pixels, preprocessing.get("mri_percentiles", [1, 99]))

    # A constant slice has no intensity range and must not cause division by zero.
    if upper <= lower:
        normalized = np.zeros_like(pixels)
    else:
        normalized = np.clip((pixels - lower) / (upper - lower), 0, 1)

    # MONOCHROME1 displays lower values as brighter, so reverse the scaled intensities.
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        normalized = 1 - normalized

    # Add the grayscale channel axis: [height, width] -> [1, height, width].
    return torch.from_numpy(np.asarray(normalized, dtype=np.float32)).unsqueeze(0)


def load_image(path, modality, preprocessing):
    if modality in ("ct", "mri"):
        return dicom_pixels(pydicom.dcmread(path), modality, preprocessing)

    # PNG/JPEG inputs use their integer dtype range, not the DICOM normalization rules.
    image = read_image(str(path), mode=ImageReadMode.GRAY)
    if image.is_floating_point():
        raise ValueError("Expected an integer PNG/JPEG image")

    return image.to(torch.float32) / torch.iinfo(image.dtype).max
