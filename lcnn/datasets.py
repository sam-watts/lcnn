import glob
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M
from lcnn.utils import apply_gaussian_heatmap


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split, image_dir=None):
        self.rootdir = rootdir
        self.image_dir = Path(image_dir) if image_dir else None
        filelist = list(Path(rootdir).joinpath(split).glob("*.npz"))
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

        # Store image normalization parameters from config
        # This ensures they're available in multiprocessing workers
        self.image_mean = np.array(M.image.mean)
        self.image_stddev = np.array(M.image.stddev)
        self.n_stc_posl = M.n_stc_posl
        self.n_stc_negl = M.n_stc_negl
        self.use_cood = M.use_cood
        self.use_slop = M.use_slop
        self.jmap_gaussian_sigma = getattr(M, "jmap_gaussian_sigma", 0)

    def __len__(self):
        return len(self.filelist)
    
    def get_image_path(self, index):
        if self.image_dir:
            return self.image_dir / f"{self.filelist[index].stem}.png"
        else:
            return self.filelist[index].with_suffix(".png")
        
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
        with np.load(self.filelist[idx]) as npz:
            jmap = npz["jmap"]
            joff = npz["joff"]
            lmap = npz["lmap"]

            # Apply Gaussian heatmap to junction map if sigma > 0
            if self.jmap_gaussian_sigma > 0:
                jmap = apply_gaussian_heatmap(
                    jmap, joff, sigma=self.jmap_gaussian_sigma
                )

            target = {
                "jmap": torch.from_numpy(jmap).float(),
                "joff": torch.from_numpy(joff).float(),
                "lmap": torch.from_numpy(lmap).float(),
            }
            lpos = np.random.permutation(npz["lpos"])[: self.n_stc_posl]
            lneg = np.random.permutation(npz["lneg"])[: self.n_stc_negl]
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
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
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
