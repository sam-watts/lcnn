import glob
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import numpy.linalg as LA
import torch
from scipy.ndimage import rotate as ndimage_rotate
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M

DATA_SOURCES = {
    "gfrid": {
        "label_dir": "/home/swatts/RoofMapNet/processed/gfrid/",
        "image_dir": "/home/swatts/datasets/gfrid/images/",
        "images_split": True
    },
    "gfrid_roof_centered": {
        "label_dir": "/home/swatts/RoofMapNet/processed/gfrid_roof_centered/",
        "image_dir": "/home/swatts/datasets/gfrid/roof_centered/edge_centerlines/images/",
        "images_split": True
    },
    "rid2": {
        "label_dir": "/home/swatts/RoofMapNet/processed/rid2",
        "image_dir": "/home/swatts/datasets/roof_information_dataset_2/images/",
        "images_split": False
    },
    "roofmapnet": {
        "label_dir": "/home/swatts/RoofMapNet/processed/roofmapnet",
        "image_dir": "/home/swatts/RoofMapNet/data/images",
        "images_split": False
    }
}

def _clip_segment(y0, x0, y1, x1, lo, hi):
    """Liang-Barsky clip of segment (y0,x0)-(y1,x1) to [lo,hi]x[lo,hi].

    Returns (y0,x0,y1,x1) of clipped segment, or None if fully outside.
    """
    dy = y1 - y0
    dx = x1 - x0
    t0, t1 = 0.0, 1.0
    for p, q in [(-dy, y0 - lo), (dy, hi - y0),
                 (-dx, x0 - lo), (dx, hi - x0)]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if t0 > t1:
            return None
    return (y0 + t0 * dy, x0 + t0 * dx, y0 + t1 * dy, x0 + t1 * dx)


def _deduplicate_lines(lpos, Lpos):
    """Remove duplicate lines from lpos and Lpos.

    Lines are considered duplicates if they share the same pair of junction
    indices (in either order).  Both the coordinate array (lpos) and the
    index array (Lpos) are filtered identically.

    Args:
        lpos: [M, 2, 3] line endpoint coordinates (y, x, type).
        Lpos: [M, 2]    junction index pairs.

    Returns:
        (lpos_unique, Lpos_unique) with duplicates removed.
    """
    if len(Lpos) == 0:
        return lpos, Lpos

    # Canonical key: sorted junction-index pair
    canonical = np.sort(Lpos, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original order

    return lpos[unique_idx], Lpos[unique_idx]


class WireframeDataset(Dataset):
    def __init__(self, split, data_sources):
        self.filelist = []
        for source in data_sources:
            if source not in DATA_SOURCES:
                raise ValueError(f"Unknown data source: {source}")

            source_config = DATA_SOURCES[source]
            label_dir = Path(source_config["label_dir"])
            image_dir = Path(source_config["image_dir"]) if source_config["image_dir"] else None
            images_split = source_config["images_split"]

            source_files = list(label_dir.joinpath(split).glob("*.npz"))
            source_files.sort()

            for lpath in source_files:
                if image_dir:
                    if images_split:
                        ipath = image_dir / split / f"{lpath.stem}.png"
                    else:
                        ipath = image_dir / f"{lpath.stem}.png"
                else:
                    ipath = lpath.with_suffix(".png")

                self.filelist.append({
                    "label": lpath,
                    "image": ipath
                })

        print(f"n{split}:", len(self.filelist))
        self.split = split

        # Store image normalization parameters from config
        # This ensures they're available in multiprocessing workers
        self.image_mean = np.array(M.image.mean)
        self.image_stddev = np.array(M.image.stddev)
        self.n_stc_posl = M.n_stc_posl
        self.n_stc_negl = M.n_stc_negl
        self.use_cood = M.use_cood
        self.use_slop = M.use_slop

        # Augmentation params (only active during training)
        aug = getattr(M, "augmentation", None)
        if split == "train" and aug is not None:
            self.flip_prob = getattr(aug, "flip_prob", 0.0)
            self.rotate_prob = getattr(aug, "rotate_prob", 0.0)
            self.rotate_max_angle = getattr(aug, "rotate_max_angle", 90)
        else:
            self.flip_prob = 0.0
            self.rotate_prob = 0.0
            self.rotate_max_angle = 0

    def __len__(self):
        return len(self.filelist)

    def get_image_path(self, index):
        return self.filelist[index]["image"]

    def _augment(self, image, npz):
        """Apply augmentations to image and label arrays in-place.

        Args:
            image: [3, H, W] normalized image
            npz: dict of numpy arrays (jmap, joff, lmap, junc, lpos, lneg, Lpos, Lneg)
        Returns:
            image, npz (augmented copies)
        """
        jmap = npz["jmap"]    # [J, H', W']
        joff = npz["joff"]    # [J, 2, H', W']
        lmap = npz["lmap"]    # [H', W']
        junc = npz["junc"]    # [Na, 3]  (y, x, type)
        lpos = npz["lpos"]    # [N, 2, 3]
        lneg = npz["lneg"]    # [N, 2, 3]
        Lpos = npz["Lpos"]    # [M, 2]
        Lneg = npz["Lneg"]    # [M, 2]

        hmap_w = jmap.shape[2]  # 128
        hmap_h = jmap.shape[1]  # 128

        # --- Horizontal flip ---
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=2).copy()
            jmap = np.flip(jmap, axis=2).copy()
            lmap = np.flip(lmap, axis=1).copy()
            joff = np.flip(joff, axis=3).copy()
            joff[:, 1] *= -1  # negate dx channel

            junc = junc.copy()
            junc[:, 1] = (hmap_w - 1) - junc[:, 1]

            if len(lpos) > 0:
                lpos = lpos.copy()
                lpos[:, :, 1] = (hmap_w - 1) - lpos[:, :, 1]
            if len(lneg) > 0:
                lneg = lneg.copy()
                lneg[:, :, 1] = (hmap_w - 1) - lneg[:, :, 1]

        # --- Vertical flip ---
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            jmap = np.flip(jmap, axis=1).copy()
            lmap = np.flip(lmap, axis=0).copy()
            joff = np.flip(joff, axis=2).copy()
            joff[:, 0] *= -1  # negate dy channel

            junc = junc.copy()
            junc[:, 0] = (hmap_h - 1) - junc[:, 0]
            if len(lpos) > 0:
                lpos = lpos.copy()
                lpos[:, :, 0] = (hmap_h - 1) - lpos[:, :, 0]
            if len(lneg) > 0:
                lneg = lneg.copy()
                lneg[:, :, 0] = (hmap_h - 1) - lneg[:, :, 0]

        # --- Rotation ---
        if random.random() < self.rotate_prob:
            angle = random.uniform(0, self.rotate_max_angle)
            if angle < 0.1:
                # Skip trivially small rotations
                return image, {
                    "jmap": jmap, "joff": joff, "lmap": lmap,
                    "junc": junc, "lpos": lpos, "lneg": lneg,
                    "Lpos": Lpos, "Lneg": Lneg,
                }

            rad = np.deg2rad(angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            # Rotate spatial arrays
            # scipy.ndimage.rotate rotates CCW for positive angles
            rot_kw = dict(reshape=False, order=1, mode="constant", cval=0)

            image = ndimage_rotate(image, angle, axes=(1, 2), **rot_kw)
            jmap = ndimage_rotate(jmap, angle, axes=(1, 2), **rot_kw)
            lmap = ndimage_rotate(lmap, angle, axes=(0, 1), **rot_kw)

            # Rotate joff spatial maps, then rotate offset vectors
            joff = ndimage_rotate(joff, angle, axes=(2, 3), **rot_kw)
            dy_old = joff[:, 0].copy()
            dx_old = joff[:, 1].copy()
            joff[:, 0] = cos_a * dy_old - sin_a * dx_old
            joff[:, 1] = sin_a * dy_old + cos_a * dx_old

            # Rotate coordinate arrays about heatmap center
            cy, cx = (hmap_w - 1) / 2.0, (hmap_w - 1) / 2.0

            def rotate_coords(coords):
                """Rotate (y, x) columns in-place. coords shape: [..., 3] with (y, x, type)."""
                y = coords[..., 0] - cy
                x = coords[..., 1] - cx
                coords[..., 0] = cos_a * y - sin_a * x + cy
                coords[..., 1] = sin_a * y + cos_a * x + cx

            junc = junc.copy()
            rotate_coords(junc)
            if len(lpos) > 0:
                lpos = lpos.copy()
                rotate_coords(lpos)
            if len(lneg) > 0:
                lneg = lneg.copy()
                rotate_coords(lneg)

            # Remove out-of-bounds junctions, remap Lpos/Lneg indices
            bound = hmap_w - 1
            valid_mask = (
                (junc[:, 0] >= 0) & (junc[:, 0] < hmap_w) &
                (junc[:, 1] >= 0) & (junc[:, 1] < hmap_w)
            )
            old_to_new = np.full(len(junc), -1, dtype=np.int64)
            new_indices = np.where(valid_mask)[0]
            old_to_new[new_indices] = np.arange(len(new_indices))
            junc = junc[valid_mask]

            def remap_lines(L):
                if len(L) == 0:
                    return L
                new_L = old_to_new[L]
                keep = (new_L[:, 0] >= 0) & (new_L[:, 1] >= 0)
                return new_L[keep]

            Lpos = remap_lines(Lpos)
            Lneg = remap_lines(Lneg)

            # Clip lpos/lneg line endpoints to image boundary
            def clip_lines(lines):
                if lines.ndim != 3 or len(lines) == 0:
                    return np.zeros((0, 2, 3), dtype=lines.dtype)
                out = lines.copy()
                keep = np.ones(len(out), dtype=bool)
                for i in range(len(out)):
                    result = _clip_segment(
                        out[i, 0, 0], out[i, 0, 1],
                        out[i, 1, 0], out[i, 1, 1], 0, bound)
                    if result is None:
                        keep[i] = False
                    else:
                        out[i, 0, 0], out[i, 0, 1] = result[0], result[1]
                        out[i, 1, 0], out[i, 1, 1] = result[2], result[3]
                return out[keep]

            lpos = clip_lines(lpos)
            lneg = clip_lines(lneg)

            # Add new junctions at clipped boundary endpoints so that
            # every line endpoint has a corresponding junction dot.
            boundary_juncs = []
            for lines in [lpos, lneg]:
                if len(lines) == 0:
                    continue
                for ei in [0, 1]:  # each endpoint
                    pts = lines[:, ei, :]  # [N, 3] (y, x, type)
                    on_edge = (
                        (pts[:, 0] <= 0) | (pts[:, 0] >= bound) |
                        (pts[:, 1] <= 0) | (pts[:, 1] >= bound)
                    )
                    if on_edge.any():
                        boundary_juncs.append(pts[on_edge])
            if boundary_juncs:
                junc = np.concatenate([junc] + boundary_juncs, axis=0)

        return image, {
            "jmap": jmap, "joff": joff, "lmap": lmap,
            "junc": junc, "lpos": lpos, "lneg": lneg,
            "Lpos": Lpos, "Lneg": Lneg,
        }

    def __getitem__(self, idx):
        # Determine image path
        iname = self.get_image_path(idx)
        image = io.imread(iname).astype(float)[:, :, :3]

        image = (image - self.image_mean) / self.image_stddev
        image = np.rollaxis(image, 2).copy()

        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]["label"]) as npz:
            npz_data = {name: npz[name].copy() for name in
                        ["jmap", "joff", "lmap", "junc", "lpos", "lneg", "Lpos", "Lneg"]}

        # Deduplicate positive lines (preprocessing may produce duplicates)
        npz_data["lpos"], npz_data["Lpos"] = _deduplicate_lines(
            npz_data["lpos"], npz_data["Lpos"]
        )

        # Apply augmentations (no-op for val/test)
        image, npz_data = self._augment(image, npz_data)

        target = {
            name: torch.from_numpy(npz_data[name]).float()
            for name in ["jmap", "joff", "lmap"]
        }
        lpos = np.random.permutation(npz_data["lpos"])[: self.n_stc_posl]
        lneg = np.random.permutation(npz_data["lneg"])[: self.n_stc_negl]

        if lneg.shape[0] == 0 and lpos.shape[0] > 0:
            lneg = np.zeros((1, lpos.shape[1], lpos.shape[2]), dtype=np.float32)

        npos, nneg = len(lpos), len(lneg)
        lpre = np.concatenate([lpos, lneg], 0)
        for i in range(len(lpre)):
            if random.random() > 0.5:
                lpre[i] = lpre[i, ::-1]

        ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
        ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
        feat = [
            lpre[:, :, :2].reshape(-1, 4) / 128 * self.use_cood,
            ldir * self.use_slop,
            lpre[:, :, 2],
        ]
        feat = np.concatenate(feat, 1)
        meta = {
            "junc": torch.from_numpy(npz_data["junc"][:, :2]),
            "jtyp": torch.from_numpy(npz_data["junc"][:, 2]).byte(),
            "Lpos": self.adjacency_matrix(len(npz_data["junc"]), npz_data["Lpos"]),
            "Lneg": self.adjacency_matrix(len(npz_data["junc"]), npz_data["Lneg"]),
            "lpre": torch.from_numpy(lpre[:, :, :2]),
            "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
            "lpre_feat": torch.from_numpy(feat),
        }

        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )


if __name__ == "__main__":
    import click
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    @click.command()
    @click.option("--flip-prob", default=0.5, type=float, help="Horizontal flip probability")
    @click.option("--rotate-prob", default=0.5, type=float, help="Rotation probability")
    @click.option("--rotate-max-angle", default=90, type=float, help="Max rotation angle in degrees")
    @click.option("--data-sources", default="gfrid", type=str, help="Comma-separated data sources")
    @click.option("--split", default="train", type=str, help="Dataset split")
    @click.option("--n-samples", default=5, type=int, help="Number of samples to plot")
    @click.option("--output-dir", default=None, type=str, help="Output directory (default: /tmp/aug_viz)")
    def main(flip_prob, rotate_prob, rotate_max_angle, data_sources, split, n_samples, output_dir):
        """Visualize augmented dataset samples."""
        from lcnn.config import C

        # Load config to initialize M
        C.update(C.from_yaml(filename="/home/swatts/lcnn/config/wireframe.yaml"))
        M.update(C.model)

        # Override augmentation params from CLI
        if not hasattr(M, "augmentation") or M.augmentation is None:
            M.augmentation = type(M)()
        M.augmentation.flip_prob = flip_prob
        M.augmentation.rotate_prob = rotate_prob
        M.augmentation.rotate_max_angle = rotate_max_angle

        sources = [s.strip() for s in data_sources.split(",")]
        dataset = WireframeDataset(split, sources)

        if output_dir is None:
            output_dir = "/tmp/aug_viz"
        os.makedirs(output_dir, exist_ok=True)

        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        for i, idx in enumerate(indices):
            image_tensor, meta, target = dataset[idx]

            # Un-normalize image back to RGB [0, 1]
            image_np = image_tensor.numpy()  # [3, H, W]
            image_rgb = np.rollaxis(image_np, 0, 3)  # [H, W, 3]
            image_rgb = image_rgb * dataset.image_stddev + dataset.image_mean
            image_rgb = np.clip(image_rgb / 255.0, 0, 1)

            junc = meta["junc"].numpy()     # [N, 2] (y, x)
            jtyp = meta["jtyp"].numpy()     # [N]
            lmap = target["lmap"].numpy()    # [H', W']
            joff = target["joff"].numpy()    # [J, 2, H', W']

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            # Left: image with junctions and lines
            ax = axes[0]
            ax.imshow(image_rgb)
            ax.set_title(f"Sample {idx} - Image + Junctions + Lines")

            # Scale junction coords from 128x128 heatmap space to image space
            h_img, w_img = image_rgb.shape[:2]
            h_hmap = lmap.shape[0]
            scale = h_img / h_hmap

            if len(junc) > 0:
                # Apply offsets from the joff map to the integer junction positions
                junc_refined = []
                for j in junc:
                    iy, ix = int(j[0]), int(j[1])
                    # Clamp indices to heatmap bounds
                    iy = min(max(iy, 0), h_hmap - 1)
                    ix = min(max(ix, 0), h_hmap - 1)
                    
                    dy = joff[0, 0, iy, ix]
                    dx = joff[0, 1, iy, ix]
                    
                    # Refined position = bin index + center offset (0.5) + predicted offset
                    junc_refined.append([iy + 0.5 + dy, ix + 0.5 + dx])
                
                junc_refined = np.array(junc_refined)
                colors = ["red" if t == 0 else "blue" for t in jtyp]
                ax.scatter(junc_refined[:, 1] * scale, junc_refined[:, 0] * scale, c=colors, s=10, zorder=5)

            # Draw positive lines from lpre
            lpre = meta["lpre"].numpy()  # [N, 2, 2] (y, x)
            lpre_label = meta["lpre_label"].numpy()
            pos_lines = lpre[lpre_label > 0.5]
            if len(pos_lines) > 0:
                segments = []
                for line in pos_lines:
                    p0 = (line[0, 1] * scale, line[0, 0] * scale)  # (x, y)
                    p1 = (line[1, 1] * scale, line[1, 0] * scale)
                    segments.append([p0, p1])
                lc = LineCollection(segments, colors="lime", linewidths=1, alpha=0.7)
                ax.add_collection(lc)

            ax.axis("off")

            # Right: lmap heatmap
            ax = axes[1]
            ax.imshow(lmap, cmap="hot")
            ax.set_title(f"Sample {idx} - Line Heatmap")
            ax.axis("off")

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"sample_{i}_{idx}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")

        print(f"\nAll plots saved to {output_dir}")

    main()
