import math
import os.path as osp
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib.pyplot as plt


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


def gaussian_2d(shape, center, sigma=1.0):
    """Generate a 2D Gaussian kernel centered at the given position.

    Args:
        shape: (H, W) tuple for the output heatmap dimensions.
        center: (y, x) tuple for the center of the Gaussian (can be float).
        sigma: Standard deviation of the Gaussian in pixels.

    Returns:
        A numpy array of shape (H, W) with the Gaussian values, peak = 1.0.
    """
    H, W = shape
    cy, cx = center

    # Determine the effective radius (3*sigma covers 99.7% of the distribution)
    radius = int(3 * sigma + 0.5)

    # Compute bounding box to avoid computing over the entire heatmap
    y_min = max(0, int(cy - radius))
    y_max = min(H, int(cy + radius) + 1)
    x_min = max(0, int(cx - radius))
    x_max = min(W, int(cx + radius) + 1)

    if y_min >= y_max or x_min >= x_max:
        return np.zeros(shape, dtype=np.float32)

    # Create coordinate grids for the bounding box only
    y = np.arange(y_min, y_max, dtype=np.float32)
    x = np.arange(x_min, x_max, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Compute the Gaussian
    gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    # Place the Gaussian patch into the full-sized heatmap
    heatmap = np.zeros(shape, dtype=np.float32)
    heatmap[y_min:y_max, x_min:x_max] = gaussian

    return heatmap


def apply_gaussian_heatmap(jmap, joff=None, sigma=1.0):
    """Convert a binary junction heatmap to a Gaussian heatmap.

    Takes a binary junction heatmap where 1s mark junction locations and
    replaces each junction with a 2D Gaussian blob. If junction offsets
    are provided, the Gaussians are centered at the sub-pixel positions.

    Args:
        jmap: Binary junction heatmap of shape [J, H, W] where J is the
              number of junction types. Values are 0 or 1.
        joff: Optional junction offset map of shape [J, 2, H, W]. The two
              channels represent (dy, dx) sub-pixel offsets in [-0.5, 0.5].
        sigma: Standard deviation of the Gaussian in pixels.

    Returns:
        A float32 array of the same shape as jmap with Gaussian heatmaps.
        Values are in [0, 1] with peak = 1.0 at each junction center.
    """
    n_types, H, W = jmap.shape
    gaussian_jmap = np.zeros_like(jmap, dtype=np.float32)

    for t in range(n_types):
        # Find junction positions in this type channel
        ys, xs = np.where(jmap[t] > 0.5)

        for y, x in zip(ys, xs):
            # Use sub-pixel center if offsets are provided
            if joff is not None:
                cy = y + joff[t, 0, y, x] + 0.5
                cx = x + joff[t, 1, y, x] + 0.5
            else:
                cy, cx = float(y), float(x)

            # Generate Gaussian and take element-wise maximum
            g = gaussian_2d((H, W), (cy, cx), sigma=sigma)
            gaussian_jmap[t] = np.maximum(gaussian_jmap[t], g)

    return gaussian_jmap


def apply_gaussian_heatmap_torch(jmap, joff=None, sigma=1.0):
    """Torch version: Convert a binary junction heatmap to a Gaussian heatmap.

    Args:
        jmap: Binary junction heatmap tensor of shape [J, H, W].
        joff: Optional junction offset tensor of shape [J, 2, H, W].
        sigma: Standard deviation of the Gaussian in pixels.

    Returns:
        A float tensor of the same shape as jmap with Gaussian heatmaps.
    """
    jmap_np = jmap.numpy() if isinstance(jmap, torch.Tensor) else jmap
    joff_np = joff.numpy() if isinstance(joff, torch.Tensor) and joff is not None else joff

    result = apply_gaussian_heatmap(jmap_np, joff_np, sigma=sigma)

    if isinstance(jmap, torch.Tensor):
        return torch.from_numpy(result).float()
    return result


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
