import math
import os.path as osp
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib.pyplot as plt


def gaussian_2d(shape, sigma=1.0):
    """Generate a 2D Gaussian kernel.

    Args:
        shape: (height, width) of the kernel. Should be odd numbers.
        sigma: Standard deviation of the Gaussian.

    Returns:
        np.ndarray of shape (height, width) with values in [0, 1].
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, sigma):
    """Draw a 2D Gaussian on the heatmap at the given center using element-wise max.

    This follows the CenterNet/CornerNet approach where overlapping Gaussians
    are combined via max rather than sum, preventing saturation.

    Args:
        heatmap: 2D numpy array to draw on (modified in-place).
        center: (row, col) center of the Gaussian (integer pixel coordinates).
        sigma: Standard deviation of the Gaussian.

    Returns:
        The modified heatmap (same reference as input).
    """
    radius = int(3 * sigma + 0.5)  # 3-sigma covers ~99.7% of the distribution
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=sigma)

    row, col = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    top = min(row, radius)
    bottom = min(height - row - 1, radius)
    left = min(col, radius)
    right = min(width - col - 1, radius)

    if top + bottom < 0 or left + right < 0:
        return heatmap

    masked_heatmap = heatmap[
        row - top : row + bottom + 1, col - left : col + right + 1
    ]
    masked_gaussian = gaussian[
        radius - top : radius + bottom + 1, radius - left : radius + right + 1
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


def generate_gaussian_jmap(jmap_shape, junctions, sigma=1.0):
    """Generate a junction heatmap with Gaussian distributions.

    Args:
        jmap_shape: (H, W) shape of the heatmap.
        junctions: List of (row, col) junction coordinates (can be sub-pixel).
        sigma: Standard deviation of the Gaussian.

    Returns:
        np.ndarray of shape (H, W) with Gaussian heatmap values in [0, 1].
    """
    heatmap = np.zeros(jmap_shape, dtype=np.float32)
    for junc in junctions:
        row, col = int(junc[0]), int(junc[1])
        if 0 <= row < jmap_shape[0] and 0 <= col < jmap_shape[1]:
            draw_gaussian(heatmap, (row, col), sigma)
    return heatmap


class benchmark(object):
    def __init__(self, msg, enable=True, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt
        self.enable = enable

    def __enter__(self):
        if self.enable:
            self.start = timer()
        return self

    def __exit__(self, *args):
        if self.enable:
            t = timer() - self.start
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
            self.time = t


def quiver(x, y, ax):
    ax.set_xlim(0, x.shape[1])
    ax.set_ylim(x.shape[0], 0)
    ax.quiver(
        x,
        y,
        units="xy",
        angles="xy",
        scale_units="xy",
        scale=1,
        minlength=0.01,
        width=0.1,
        color="b",
    )


def recursive_to(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], torch.Tensor):
                input[name] = input[name].to(device)
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item, device)
        return input
    assert False


def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def argsort2d(arr):
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]


def __parallel_handle(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(X)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]
