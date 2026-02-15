#!/usr/bin/env python3
"""Compute per-channel (RGB) mean and std of training images for aerial roof datasets.

Uses Welford's online algorithm for numerically stable single-pass computation,
processing images one at a time to keep memory usage low.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image


def get_train_stems(label_dir: str) -> list[str]:
    """Get sorted list of stems from .npz files in the label directory."""
    label_path = Path(label_dir)
    stems = sorted(p.stem for p in label_path.glob("*.npz"))
    return stems


def find_image(image_dir: str, stem: str) -> str | None:
    """Find image file for a given stem, trying .png then .jpg."""
    for ext in (".png", ".jpg"):
        path = os.path.join(image_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def compute_stats(image_dir: str, label_dir: str, dataset_name: str):
    """Compute per-channel mean and std using Welford's online algorithm.

    Returns (mean, std, n_images) where mean and std are arrays of shape (3,)
    in [0, 255] scale, and also (sum_pixels, sum_sq_pixels, total_pixel_count)
    for combining across datasets.
    """
    stems = get_train_stems(label_dir)
    print(f"  {dataset_name}: found {len(stems)} train stems, loading images...")

    # Welford's: track count, mean, M2 per channel
    # Here we accumulate over ALL pixels (not per-image means)
    # For N pixels total across all images:
    #   channel_sum[c] = sum of all pixel values in channel c
    #   channel_sum_sq[c] = sum of squared pixel values in channel c
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    n_images = 0
    missing = 0

    for i, stem in enumerate(stems):
        img_path = find_image(image_dir, stem)
        if img_path is None:
            missing += 1
            continue

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float64)  # (H, W, 3)
        h, w, c = arr.shape
        n_px = h * w

        # Reshape to (n_px, 3) and accumulate
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sum_sq += (flat ** 2).sum(axis=0)
        total_pixels += n_px
        n_images += 1

        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(stems)} images...")

    if missing > 0:
        print(f"  WARNING: {missing} images not found")

    mean = channel_sum / total_pixels
    std = np.sqrt(channel_sum_sq / total_pixels - mean ** 2)

    return mean, std, n_images, channel_sum, channel_sum_sq, total_pixels


def main():
    datasets = [
        {
            "name": "gfrid",
            "image_dir": "/home/swatts/datasets/gfrid/images/train/",
            "label_dir": "/home/swatts/RoofMapNet/processed/gfrid/train/",
        },
        {
            "name": "roofmapnet",
            "image_dir": "/home/swatts/RoofMapNet/data/images/",
            "label_dir": "/home/swatts/RoofMapNet/processed/roofmapnet/train/",
        },
        {
            "name": "rid2",
            "image_dir": "/home/swatts/datasets/roof_information_dataset_2/images/",
            "label_dir": "/home/swatts/RoofMapNet/processed/rid2/train/",
        },
    ]

    results = {}
    combined_sum = np.zeros(3, dtype=np.float64)
    combined_sum_sq = np.zeros(3, dtype=np.float64)
    combined_pixels = 0
    combined_images = 0

    print("=" * 60)
    print("Computing per-channel RGB statistics for training images")
    print("=" * 60)

    for ds in datasets:
        mean, std, n_images, ch_sum, ch_sum_sq, total_px = compute_stats(
            ds["image_dir"], ds["label_dir"], ds["name"]
        )
        results[ds["name"]] = (mean, std, n_images)
        combined_sum += ch_sum
        combined_sum_sq += ch_sum_sq
        combined_pixels += total_px
        combined_images += n_images

    # Combined stats
    combined_mean = combined_sum / combined_pixels
    combined_std = np.sqrt(combined_sum_sq / combined_pixels - combined_mean ** 2)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (pixel values in [0, 255] scale)")
    print("=" * 60)

    for name, (mean, std, n_images) in results.items():
        print(f"\nDataset: {name} ({n_images} images)")
        print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    print(f"\nDataset: combined ({combined_images} images)")
    print(f"  mean: [{combined_mean[0]:.4f}, {combined_mean[1]:.4f}, {combined_mean[2]:.4f}]")
    print(f"  std:  [{combined_std[0]:.4f}, {combined_std[1]:.4f}, {combined_std[2]:.4f}]")

    # Also print normalized [0, 1] scale for convenience
    print("\n" + "=" * 60)
    print("RESULTS (normalized to [0, 1] scale)")
    print("=" * 60)

    for name, (mean, std, n_images) in results.items():
        m = mean / 255.0
        s = std / 255.0
        print(f"\nDataset: {name} ({n_images} images)")
        print(f"  mean: [{m[0]:.6f}, {m[1]:.6f}, {m[2]:.6f}]")
        print(f"  std:  [{s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}]")

    cm = combined_mean / 255.0
    cs = combined_std / 255.0
    print(f"\nDataset: combined ({combined_images} images)")
    print(f"  mean: [{cm[0]:.6f}, {cm[1]:.6f}, {cm[2]:.6f}]")
    print(f"  std:  [{cs[0]:.6f}, {cs[1]:.6f}, {cs[2]:.6f}]")


if __name__ == "__main__":
    main()
