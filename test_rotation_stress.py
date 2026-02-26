#!/usr/bin/env python3
"""Stress tests: combined flip+rotation, edge cases, dtype checks."""

import sys
import random
import numpy as np
import skimage.draw

from lcnn.config import M
from lcnn.box import Box

M.update({
    "image": {"mean": [0.0, 0.0, 0.0], "stddev": [1.0, 1.0, 1.0]},
    "n_stc_posl": 300,
    "n_stc_negl": 40,
    "use_cood": 0,
    "use_slop": 0,
    "augmentation": {
        "flip_prob": 0.5,
        "rotate_prob": 1.0,
        "rotate_max_angle": 90,
    },
})

from lcnn.datasets import WireframeDataset


def make_synthetic(hmap=128, n_junc=10, n_lines=8):
    """Random wireframe inside [margin, hmap-margin]."""
    margin = 20
    junctions = np.zeros((n_junc, 3), dtype=np.float32)
    junctions[:, 0] = np.random.uniform(margin, hmap - margin, n_junc)
    junctions[:, 1] = np.random.uniform(margin, hmap - margin, n_junc)
    # type = 0

    # Random positive lines
    idx = np.random.choice(n_junc, size=(n_lines, 2), replace=True)
    # Remove self-loops
    idx = idx[idx[:, 0] != idx[:, 1]]
    Lpos = idx.astype(np.int64)
    lpos = np.stack([junctions[Lpos[:, 0]], junctions[Lpos[:, 1]]], axis=1)

    # Random negative lines
    n_neg = min(5, n_junc * (n_junc - 1) // 2)
    neg_idx = np.random.choice(n_junc, size=(n_neg, 2), replace=True)
    neg_idx = neg_idx[neg_idx[:, 0] != neg_idx[:, 1]]
    Lneg = neg_idx.astype(np.int64)
    lneg = np.stack([junctions[Lneg[:, 0]], junctions[Lneg[:, 1]]], axis=1) if len(Lneg) > 0 else np.zeros((0, 2, 3), dtype=np.float32)

    jmap = np.zeros((1, hmap, hmap), dtype=np.float32)
    joff = np.zeros((1, 2, hmap, hmap), dtype=np.float32)
    lmap = np.zeros((hmap, hmap), dtype=np.float32)

    for v in junctions:
        iy, ix = int(v[0]), int(v[1])
        if 0 <= iy < hmap and 0 <= ix < hmap:
            jmap[0, iy, ix] = 1
            joff[0, 0, iy, ix] = v[0] - iy - 0.5
            joff[0, 1, iy, ix] = v[1] - ix - 0.5

    for line in lpos:
        v0 = (max(0, min(hmap-1, int(round(line[0, 0])))),
               max(0, min(hmap-1, int(round(line[0, 1])))))
        v1 = (max(0, min(hmap-1, int(round(line[1, 0])))),
               max(0, min(hmap-1, int(round(line[1, 1])))))
        rr, cc, val = skimage.draw.line_aa(v0[0], v0[1], v1[0], v1[1])
        mask = (rr >= 0) & (rr < hmap) & (cc >= 0) & (cc < hmap)
        lmap[rr[mask], cc[mask]] = np.maximum(lmap[rr[mask], cc[mask]], val[mask])

    image = np.random.randn(3, 512, 512).astype(np.float64)
    return image, {
        "jmap": jmap, "joff": joff, "lmap": lmap,
        "junc": junctions, "lpos": lpos, "lneg": lneg,
        "Lpos": Lpos, "Lneg": Lneg,
    }


def make_ds():
    ds = object.__new__(WireframeDataset)
    ds.flip_prob = 0.5
    ds.rotate_prob = 1.0
    ds.rotate_max_angle = 90
    return ds


def test_combined_flip_rotate_consistency():
    """Run many trials with flip+rotate, check all invariants."""
    ds = make_ds()
    n_trials = 100
    for trial in range(n_trials):
        random.seed(trial)
        np.random.seed(trial)

        image, npz = make_synthetic()
        _, aug = ds._augment(image, npz)

        junc = aug["junc"]
        Lpos = aug["Lpos"]
        Lneg = aug["Lneg"]
        lpos = aug["lpos"]
        lneg = aug["lneg"]
        jmap = aug["jmap"]
        lmap = aug["lmap"]

        # 1) All junctions in bounds
        if len(junc) > 0:
            assert np.all(junc[:, 0] >= 0) and np.all(junc[:, 0] < 128), \
                f"Trial {trial}: junction y OOB"
            assert np.all(junc[:, 1] >= 0) and np.all(junc[:, 1] < 128), \
                f"Trial {trial}: junction x OOB"

        # 2) Lpos/Lneg indices valid
        for name, L in [("Lpos", Lpos), ("Lneg", Lneg)]:
            if len(L) > 0:
                assert np.all(L >= 0), f"Trial {trial}: {name} negative idx"
                assert np.all(L < len(junc)), f"Trial {trial}: {name} idx >= n_junc"

        # 3) lpos/lneg count matches Lpos/Lneg
        assert len(lpos) == len(Lpos), \
            f"Trial {trial}: lpos/Lpos mismatch: {len(lpos)} vs {len(Lpos)}"
        assert len(lneg) == len(Lneg), \
            f"Trial {trial}: lneg/Lneg mismatch: {len(lneg)} vs {len(Lneg)}"

        # 4) lpos[i] == [junc[Lpos[i,0]], junc[Lpos[i,1]]]
        for i in range(len(Lpos)):
            expected = np.stack([junc[Lpos[i, 0]], junc[Lpos[i, 1]]])
            np.testing.assert_allclose(lpos[i], expected, atol=1e-5,
                err_msg=f"Trial {trial}, lpos[{i}] mismatch")

        # 5) No NaN/Inf in heatmaps
        assert not np.any(np.isnan(jmap)), f"Trial {trial}: NaN in jmap"
        assert not np.any(np.isnan(lmap)), f"Trial {trial}: NaN in lmap"
        assert not np.any(np.isinf(jmap)), f"Trial {trial}: Inf in jmap"
        assert not np.any(np.isinf(lmap)), f"Trial {trial}: Inf in lmap"

        # 6) jmap values are 0 or 1
        unique_vals = np.unique(jmap)
        assert all(v in [0.0, 1.0] for v in unique_vals), \
            f"Trial {trial}: jmap has values other than 0/1: {unique_vals}"

        # 7) lmap values in [0, 1]
        assert np.all(lmap >= 0) and np.all(lmap <= 1), \
            f"Trial {trial}: lmap values out of [0,1]"

    print(f"  PASS  test_combined_flip_rotate_consistency ({n_trials} trials)")


def test_no_rotation_when_prob_zero():
    """With rotate_prob=0, coordinate data should be unchanged (except flips)."""
    ds = make_ds()
    ds.rotate_prob = 0.0
    ds.flip_prob = 0.0

    for trial in range(10):
        random.seed(trial + 1000)
        np.random.seed(trial + 1000)
        image, npz = make_synthetic()
        _, aug = ds._augment(image, npz)

        np.testing.assert_array_equal(aug["junc"], npz["junc"])
        np.testing.assert_array_equal(aug["Lpos"], npz["Lpos"])
        np.testing.assert_array_equal(aug["lpos"], npz["lpos"])

    print("  PASS  test_no_rotation_when_prob_zero")


def test_empty_lines():
    """Handle case with 0 positive or 0 negative lines gracefully."""
    ds = make_ds()
    ds.rotate_max_angle = 45

    hmap = 128
    junctions = np.array([[60, 60, 0], [60, 70, 0]], dtype=np.float32)

    # Case 1: No positive lines
    npz1 = {
        "jmap": np.zeros((1, hmap, hmap), dtype=np.float32),
        "joff": np.zeros((1, 2, hmap, hmap), dtype=np.float32),
        "lmap": np.zeros((hmap, hmap), dtype=np.float32),
        "junc": junctions.copy(),
        "lpos": np.zeros((0, 2, 3), dtype=np.float32),
        "lneg": np.zeros((0, 2, 3), dtype=np.float32),
        "Lpos": np.zeros((0, 2), dtype=np.int64),
        "Lneg": np.zeros((0, 2), dtype=np.int64),
    }
    image = np.random.randn(3, 512, 512)

    random.seed(42)
    np.random.seed(42)
    _, aug = ds._augment(image, npz1)
    assert aug["lpos"].shape[0] == 0
    assert aug["Lpos"].shape[0] == 0
    assert aug["lneg"].shape[0] == 0

    # Case 2: Has positive, no negative
    Lpos = np.array([[0, 1]], dtype=np.int64)
    lpos = np.stack([junctions[Lpos[:, 0]], junctions[Lpos[:, 1]]], axis=1)
    npz2 = {
        "jmap": np.zeros((1, hmap, hmap), dtype=np.float32),
        "joff": np.zeros((1, 2, hmap, hmap), dtype=np.float32),
        "lmap": np.zeros((hmap, hmap), dtype=np.float32),
        "junc": junctions.copy(),
        "lpos": lpos,
        "lneg": np.zeros((0, 2, 3), dtype=np.float32),
        "Lpos": Lpos,
        "Lneg": np.zeros((0, 2), dtype=np.int64),
    }

    random.seed(42)
    np.random.seed(42)
    _, aug = ds._augment(image, npz2)
    assert len(aug["lpos"]) == len(aug["Lpos"])
    assert aug["lneg"].shape[0] == 0

    print("  PASS  test_empty_lines")


def test_adjacency_matrix_valid():
    """The adjacency_matrix method should work with the augmented data."""
    import torch
    ds = make_ds()

    for trial in range(20):
        random.seed(trial + 2000)
        np.random.seed(trial + 2000)
        image, npz = make_synthetic()
        _, aug = ds._augment(image, npz)

        n = len(aug["junc"])
        for name in ["Lpos", "Lneg"]:
            L = aug[name]
            # This should not raise
            mat = ds.adjacency_matrix(n, L)
            assert mat.shape == (n + 1, n + 1), \
                f"Trial {trial}: adjacency matrix shape wrong"

    print("  PASS  test_adjacency_matrix_valid")


if __name__ == "__main__":
    tests = [
        test_combined_flip_rotate_consistency,
        test_no_rotation_when_prob_zero,
        test_empty_lines,
        test_adjacency_matrix_valid,
    ]

    print(f"\nRunning {len(tests)} stress tests ...\n")
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
    else:
        print("All stress tests passed!")
        sys.exit(0)
