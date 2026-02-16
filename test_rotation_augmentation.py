#!/usr/bin/env python3
"""Verify the rotation augmentation fix in lcnn/datasets.py.

This test creates synthetic wireframe data (junctions + lines) and runs
the rotation augmentation at various angles, checking for the two known
bugs and other consistency invariants:

  Bug 1 – Lines missing after rotation
  Bug 2 – Excessive / invalid boundary points

Run:
    python test_rotation_augmentation.py
"""

import sys
import random
import numpy as np
import skimage.draw

# ---------------------------------------------------------------------------
# Minimal stubs so we can import WireframeDataset._augment without needing
# the real config or filesystem.
# ---------------------------------------------------------------------------
from lcnn.config import M
from lcnn.box import Box

# Set up minimal M config for the augmentation code
M.update({
    "image": {"mean": [0.0, 0.0, 0.0], "stddev": [1.0, 1.0, 1.0]},
    "n_stc_posl": 300,
    "n_stc_negl": 40,
    "use_cood": 0,
    "use_slop": 0,
    "augmentation": {
        "flip_prob": 0.0,       # disable flips for rotation tests
        "rotate_prob": 1.0,     # always rotate
        "rotate_max_angle": 90,
    },
})

from lcnn.datasets import WireframeDataset


# ---------------------------------------------------------------------------
# Helper: build synthetic wireframe data
# ---------------------------------------------------------------------------

def make_synthetic_data(hmap_h=128, hmap_w=128, img_h=512, img_w=512):
    """Create a simple square wireframe in the centre of the heatmap."""

    # 4 junctions forming a square, well inside the image
    margin = 30
    junctions = np.array([
        [margin,        margin,        0],   # top-left
        [margin,        hmap_w-margin, 0],   # top-right
        [hmap_h-margin, hmap_w-margin, 0],   # bottom-right
        [hmap_h-margin, margin,        0],   # bottom-left
    ], dtype=np.float32)

    # 4 lines forming the square (index pairs)
    Lpos = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
    ], dtype=np.int64)

    # Build coordinate-based positive lines from junctions + Lpos
    lpos = np.stack([junctions[Lpos[:, 0]], junctions[Lpos[:, 1]]], axis=1)

    # A few negative lines (diagonals)
    Lneg = np.array([
        [0, 2], [1, 3],
    ], dtype=np.int64)
    lneg = np.stack([junctions[Lneg[:, 0]], junctions[Lneg[:, 1]]], axis=1)

    # Build heatmaps
    jmap = np.zeros((1, hmap_h, hmap_w), dtype=np.float32)
    joff = np.zeros((1, 2, hmap_h, hmap_w), dtype=np.float32)
    lmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)

    for v in junctions:
        iy, ix = int(v[0]), int(v[1])
        jmap[0, iy, ix] = 1
        joff[0, 0, iy, ix] = v[0] - iy - 0.5
        joff[0, 1, iy, ix] = v[1] - ix - 0.5

    for line in lpos:
        v0 = (int(round(line[0, 0])), int(round(line[0, 1])))
        v1 = (int(round(line[1, 0])), int(round(line[1, 1])))
        rr, cc, value = skimage.draw.line_aa(v0[0], v0[1], v1[0], v1[1])
        mask = (rr >= 0) & (rr < hmap_h) & (cc >= 0) & (cc < hmap_w)
        lmap[rr[mask], cc[mask]] = np.maximum(lmap[rr[mask], cc[mask]], value[mask])

    image = np.random.randn(3, img_h, img_w).astype(np.float64)

    npz_data = {
        "jmap": jmap, "joff": joff, "lmap": lmap,
        "junc": junctions, "lpos": lpos, "lneg": lneg,
        "Lpos": Lpos, "Lneg": Lneg,
    }
    return image, npz_data


def make_dataset_stub():
    """Create a WireframeDataset-like object with augmentation params set."""
    ds = object.__new__(WireframeDataset)
    ds.flip_prob = 0.0
    ds.rotate_prob = 1.0
    ds.rotate_max_angle = 90
    return ds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_missing_lines():
    """Bug 1: After a moderate rotation every original line should survive
    (all junctions are well within bounds, so none should be pruned)."""
    ds = make_dataset_stub()
    n_trials = 20
    for trial in range(n_trials):
        random.seed(trial)
        np.random.seed(trial)

        image, npz = make_synthetic_data()
        n_lpos_before = len(npz["lpos"])
        n_lneg_before = len(npz["lneg"])
        n_Lpos_before = len(npz["Lpos"])

        # Force a moderate angle (< ~45 deg) so nothing goes OOB
        ds.rotate_max_angle = 30
        aug_image, aug_npz = ds._augment(image, npz)

        n_lpos_after = len(aug_npz["lpos"])
        n_Lpos_after = len(aug_npz["Lpos"])

        assert n_lpos_after == n_lpos_before, (
            f"Trial {trial}: lpos lines went from {n_lpos_before} → {n_lpos_after} "
            f"(expected no loss for small rotations with well-centred junctions)"
        )
        assert n_Lpos_after == n_Lpos_before, (
            f"Trial {trial}: Lpos went from {n_Lpos_before} → {n_Lpos_after}"
        )
    print("  PASS  test_no_missing_lines")


def test_no_excessive_boundary_points():
    """Bug 2: After rotation, junc should not grow larger than the original
    count (no phantom boundary junctions should be injected)."""
    ds = make_dataset_stub()
    n_trials = 20
    for trial in range(n_trials):
        random.seed(trial * 100)
        np.random.seed(trial * 100)

        image, npz = make_synthetic_data()
        n_junc_before = len(npz["junc"])

        ds.rotate_max_angle = 90
        _, aug_npz = ds._augment(image, npz)

        n_junc_after = len(aug_npz["junc"])

        assert n_junc_after <= n_junc_before, (
            f"Trial {trial}: junction count grew from {n_junc_before} → "
            f"{n_junc_after} (boundary junctions should not be injected)"
        )
    print("  PASS  test_no_excessive_boundary_points")


def test_all_junctions_in_bounds():
    """All junctions in the output must be within [0, hmap_size)."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 90
    n_trials = 50
    for trial in range(n_trials):
        random.seed(trial + 200)
        np.random.seed(trial + 200)

        image, npz = make_synthetic_data()
        _, aug_npz = ds._augment(image, npz)
        junc = aug_npz["junc"]

        if len(junc) == 0:
            continue

        assert np.all(junc[:, 0] >= 0) and np.all(junc[:, 0] < 128), (
            f"Trial {trial}: junction y out of bounds: "
            f"min={junc[:, 0].min():.2f}, max={junc[:, 0].max():.2f}"
        )
        assert np.all(junc[:, 1] >= 0) and np.all(junc[:, 1] < 128), (
            f"Trial {trial}: junction x out of bounds: "
            f"min={junc[:, 1].min():.2f}, max={junc[:, 1].max():.2f}"
        )
    print("  PASS  test_all_junctions_in_bounds")


def test_lpos_lneg_consistent_with_Lpos_Lneg():
    """lpos[i] must equal [junc[Lpos[i,0]], junc[Lpos[i,1]]], and likewise
    for lneg/Lneg."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 90
    n_trials = 30
    for trial in range(n_trials):
        random.seed(trial + 300)
        np.random.seed(trial + 300)

        image, npz = make_synthetic_data()
        _, aug = ds._augment(image, npz)

        junc = aug["junc"]
        Lpos = aug["Lpos"]
        lpos = aug["lpos"]
        Lneg = aug["Lneg"]
        lneg = aug["lneg"]

        # Check lpos ↔ Lpos consistency
        assert len(lpos) == len(Lpos), (
            f"Trial {trial}: lpos has {len(lpos)} rows but Lpos has {len(Lpos)}"
        )
        for i in range(len(Lpos)):
            expected = np.stack([junc[Lpos[i, 0]], junc[Lpos[i, 1]]])
            np.testing.assert_allclose(
                lpos[i], expected, atol=1e-5,
                err_msg=f"Trial {trial}, line {i}: lpos ≠ junc[Lpos]"
            )

        # Check lneg ↔ Lneg consistency
        assert len(lneg) == len(Lneg), (
            f"Trial {trial}: lneg has {len(lneg)} rows but Lneg has {len(Lneg)}"
        )
        for i in range(len(Lneg)):
            expected = np.stack([junc[Lneg[i, 0]], junc[Lneg[i, 1]]])
            np.testing.assert_allclose(
                lneg[i], expected, atol=1e-5,
                err_msg=f"Trial {trial}, line {i}: lneg ≠ junc[Lneg]"
            )
    print("  PASS  test_lpos_lneg_consistent_with_Lpos_Lneg")


def test_Lpos_indices_in_range():
    """All indices in Lpos/Lneg must be valid indices into junc."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 90
    n_trials = 30
    for trial in range(n_trials):
        random.seed(trial + 400)
        np.random.seed(trial + 400)

        image, npz = make_synthetic_data()
        _, aug = ds._augment(image, npz)

        n_junc = len(aug["junc"])
        for name in ["Lpos", "Lneg"]:
            L = aug[name]
            if len(L) == 0:
                continue
            assert np.all(L >= 0), f"Trial {trial}: {name} has negative index"
            assert np.all(L < n_junc), (
                f"Trial {trial}: {name} has index {L.max()} >= n_junc={n_junc}"
            )
    print("  PASS  test_Lpos_indices_in_range")


def test_heatmaps_consistent_with_coords():
    """After rotation, jmap should have a 1 at each junction's int position,
    and lmap should be non-zero along the positive line paths."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 45
    n_trials = 20
    for trial in range(n_trials):
        random.seed(trial + 500)
        np.random.seed(trial + 500)

        image, npz = make_synthetic_data()
        _, aug = ds._augment(image, npz)

        jmap = aug["jmap"]
        joff = aug["joff"]
        lmap = aug["lmap"]
        junc = aug["junc"]
        lpos = aug["lpos"]

        # Check each junction has a hot pixel in jmap
        for j_idx, v in enumerate(junc):
            iy, ix = int(v[0]), int(v[1])
            if 0 <= iy < 128 and 0 <= ix < 128:
                assert jmap[0, iy, ix] == 1.0, (
                    f"Trial {trial}: jmap missing junction at ({iy},{ix})"
                )
                # Check offset points back to the sub-pixel position
                dy = joff[0, 0, iy, ix]
                dx = joff[0, 1, iy, ix]
                recon_y = iy + 0.5 + dy
                recon_x = ix + 0.5 + dx
                assert abs(recon_y - v[0]) < 1e-4, (
                    f"Trial {trial}: joff y mismatch at junction {j_idx}"
                )
                assert abs(recon_x - v[1]) < 1e-4, (
                    f"Trial {trial}: joff x mismatch at junction {j_idx}"
                )

        # Check lmap has non-zero values along each positive line
        for l_idx, line in enumerate(lpos):
            v0 = (int(round(line[0, 0])), int(round(line[0, 1])))
            v1 = (int(round(line[1, 0])), int(round(line[1, 1])))
            v0 = (max(0, min(127, v0[0])), max(0, min(127, v0[1])))
            v1 = (max(0, min(127, v1[0])), max(0, min(127, v1[1])))
            rr, cc, _ = skimage.draw.line_aa(v0[0], v0[1], v1[0], v1[1])
            mask = (rr >= 0) & (rr < 128) & (cc >= 0) & (cc < 128)
            line_vals = lmap[rr[mask], cc[mask]]
            assert np.any(line_vals > 0), (
                f"Trial {trial}: lmap has no values along positive line {l_idx}"
            )
    print("  PASS  test_heatmaps_consistent_with_coords")


def test_image_shape_preserved():
    """The augmented image must keep its [3, H, W] shape."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 90
    for trial in range(10):
        random.seed(trial + 600)
        np.random.seed(trial + 600)

        image, npz = make_synthetic_data()
        aug_image, _ = ds._augment(image, npz)

        assert aug_image.shape == image.shape, (
            f"Image shape changed: {image.shape} → {aug_image.shape}"
        )
    print("  PASS  test_image_shape_preserved")


def test_large_rotation_drops_edge_junctions():
    """With junctions at the corners, a 90-degree rotation should drop some."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 90

    # Place junctions very close to corners — they should be pruned
    hmap = 128
    junctions = np.array([
        [2, 2, 0],           # near top-left corner
        [2, hmap-3, 0],      # near top-right corner
        [hmap-3, hmap-3, 0], # near bottom-right corner
        [hmap-3, 2, 0],      # near bottom-left corner
    ], dtype=np.float32)
    Lpos = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    lpos = np.stack([junctions[Lpos[:, 0]], junctions[Lpos[:, 1]]], axis=1)
    Lneg = np.array([[0, 2]], dtype=np.int64)
    lneg = np.stack([junctions[Lneg[:, 0]], junctions[Lneg[:, 1]]], axis=1)

    jmap = np.zeros((1, hmap, hmap), dtype=np.float32)
    joff = np.zeros((1, 2, hmap, hmap), dtype=np.float32)
    lmap = np.zeros((hmap, hmap), dtype=np.float32)
    for v in junctions:
        iy, ix = int(v[0]), int(v[1])
        jmap[0, iy, ix] = 1
    image = np.random.randn(3, 512, 512)

    npz_data = {
        "jmap": jmap, "joff": joff, "lmap": lmap,
        "junc": junctions, "lpos": lpos, "lneg": lneg,
        "Lpos": Lpos, "Lneg": Lneg,
    }

    some_dropped = False
    for trial in range(50):
        random.seed(trial + 700)
        np.random.seed(trial + 700)
        _, aug = ds._augment(image, npz_data)
        if len(aug["junc"]) < 4:
            some_dropped = True
            # Verify consistency still holds
            assert len(aug["lpos"]) == len(aug["Lpos"])
            assert len(aug["lneg"]) == len(aug["Lneg"])
            for L in [aug["Lpos"], aug["Lneg"]]:
                if len(L) > 0:
                    assert np.all(L >= 0)
                    assert np.all(L < len(aug["junc"]))
            break

    assert some_dropped, (
        "Expected some corner junctions to be pruned at large angles"
    )
    print("  PASS  test_large_rotation_drops_edge_junctions")


def test_zero_angle_no_change():
    """When rotate_max_angle is essentially 0, nothing should change."""
    ds = make_dataset_stub()
    ds.rotate_max_angle = 0.05  # will always be < 0.1 threshold

    random.seed(42)
    np.random.seed(42)
    image, npz = make_synthetic_data()
    _, aug = ds._augment(image, npz)

    np.testing.assert_array_equal(aug["junc"], npz["junc"])
    np.testing.assert_array_equal(aug["Lpos"], npz["Lpos"])
    np.testing.assert_array_equal(aug["lpos"], npz["lpos"])
    print("  PASS  test_zero_angle_no_change")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_no_missing_lines,
        test_no_excessive_boundary_points,
        test_all_junctions_in_bounds,
        test_lpos_lneg_consistent_with_Lpos_Lneg,
        test_Lpos_indices_in_range,
        test_heatmaps_consistent_with_coords,
        test_image_shape_preserved,
        test_large_rotation_drops_edge_junctions,
        test_zero_angle_no_change,
    ]

    print(f"\nRunning {len(tests)} rotation augmentation tests ...\n")
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test_fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
