"""Tests for Gaussian heatmap generation and related utilities."""

import numpy as np
import pytest
import torch

from lcnn.utils import (
    apply_gaussian_heatmap,
    apply_gaussian_heatmap_torch,
    gaussian_2d,
)


class TestGaussian2D:
    """Tests for the gaussian_2d function."""

    def test_peak_at_center(self):
        """Gaussian should have peak value of 1.0 at the specified center."""
        heatmap = gaussian_2d((128, 128), (64.0, 64.0), sigma=2.0)
        assert heatmap[64, 64] == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self):
        """Gaussian should be symmetric around the center."""
        heatmap = gaussian_2d((128, 128), (64.0, 64.0), sigma=2.0)
        # Check symmetry in all four directions
        assert heatmap[63, 64] == pytest.approx(heatmap[65, 64], abs=1e-6)
        assert heatmap[64, 63] == pytest.approx(heatmap[64, 65], abs=1e-6)
        assert heatmap[63, 63] == pytest.approx(heatmap[65, 65], abs=1e-6)

    def test_decay(self):
        """Values should decay with distance from the center."""
        heatmap = gaussian_2d((128, 128), (64.0, 64.0), sigma=2.0)
        assert heatmap[64, 64] > heatmap[65, 64] > heatmap[66, 64] > heatmap[67, 64]

    def test_output_shape(self):
        """Output should match the requested shape."""
        for h, w in [(128, 128), (64, 64), (100, 200)]:
            heatmap = gaussian_2d((h, w), (h // 2, w // 2), sigma=1.0)
            assert heatmap.shape == (h, w)

    def test_output_dtype(self):
        """Output should be float32."""
        heatmap = gaussian_2d((128, 128), (64.0, 64.0), sigma=2.0)
        assert heatmap.dtype == np.float32

    def test_values_in_range(self):
        """All values should be in [0, 1]."""
        heatmap = gaussian_2d((128, 128), (30.0, 50.0), sigma=3.0)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_corner_center(self):
        """Gaussian at corner should still produce valid output."""
        heatmap = gaussian_2d((128, 128), (0.0, 0.0), sigma=2.0)
        assert heatmap[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert heatmap.max() == pytest.approx(1.0, abs=1e-6)

    def test_edge_center(self):
        """Gaussian at edge should produce valid output."""
        heatmap = gaussian_2d((128, 128), (0.0, 64.0), sigma=2.0)
        assert heatmap[0, 64] == pytest.approx(1.0, abs=1e-6)

    def test_out_of_bounds_center(self):
        """Gaussian far outside should produce near-zero heatmap."""
        heatmap = gaussian_2d((128, 128), (-100.0, -100.0), sigma=1.0)
        assert heatmap.max() < 1e-10

    def test_subpixel_center(self):
        """Gaussian should work with floating-point center positions."""
        heatmap = gaussian_2d((128, 128), (64.3, 32.7), sigma=2.0)
        assert heatmap.max() > 0.0
        # Peak is at the sub-pixel center, so nearby integer pixels
        # should have high but not necessarily 1.0 values
        assert heatmap.max() == pytest.approx(1.0, abs=0.05)

    def test_sigma_affects_spread(self):
        """Larger sigma should produce wider spread."""
        h_small = gaussian_2d((128, 128), (64.0, 64.0), sigma=1.0)
        h_large = gaussian_2d((128, 128), (64.0, 64.0), sigma=3.0)
        # At 5 pixels away, larger sigma should have higher value
        assert h_large[64, 69] > h_small[64, 69]

    def test_sum_increases_with_sigma(self):
        """Total sum should increase with sigma (more spread)."""
        s1 = gaussian_2d((128, 128), (64.0, 64.0), sigma=1.0).sum()
        s2 = gaussian_2d((128, 128), (64.0, 64.0), sigma=2.0).sum()
        s3 = gaussian_2d((128, 128), (64.0, 64.0), sigma=3.0).sum()
        assert s1 < s2 < s3


class TestApplyGaussianHeatmap:
    """Tests for the apply_gaussian_heatmap function."""

    def test_single_junction(self):
        """A single junction should produce a single Gaussian blob."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        jmap[0, 64, 64] = 1.0
        result = apply_gaussian_heatmap(jmap, sigma=2.0)
        assert result.shape == (1, 128, 128)
        assert result[0, 64, 64] == pytest.approx(1.0, abs=1e-5)
        # Neighboring pixels should be non-zero
        assert result[0, 65, 64] > 0.0
        assert result[0, 64, 65] > 0.0

    def test_multiple_junctions(self):
        """Multiple junctions should produce overlapping Gaussians (max)."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        jmap[0, 30, 30] = 1.0
        jmap[0, 90, 90] = 1.0
        result = apply_gaussian_heatmap(jmap, sigma=2.0)
        # Both centers should be 1.0
        assert result[0, 30, 30] == pytest.approx(1.0, abs=1e-5)
        assert result[0, 90, 90] == pytest.approx(1.0, abs=1e-5)

    def test_with_offset(self):
        """Sub-pixel offsets should shift the Gaussian center."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        jmap[0, 64, 64] = 1.0
        joff = np.zeros((1, 2, 128, 128), dtype=np.float32)
        joff[0, 0, 64, 64] = 0.3  # dy offset
        joff[0, 1, 64, 64] = 0.2  # dx offset
        result = apply_gaussian_heatmap(jmap, joff=joff, sigma=2.0)
        assert result.shape == (1, 128, 128)
        # Peak should still be close to 1.0
        assert result.max() > 0.9

    def test_empty_jmap(self):
        """Empty junction map should produce zero output."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        result = apply_gaussian_heatmap(jmap, sigma=2.0)
        assert result.sum() == 0.0

    def test_multiple_types(self):
        """Should handle multiple junction types independently."""
        jmap = np.zeros((2, 128, 128), dtype=np.float32)
        jmap[0, 30, 30] = 1.0
        jmap[1, 90, 90] = 1.0
        result = apply_gaussian_heatmap(jmap, sigma=2.0)
        assert result.shape == (2, 128, 128)
        # Type 0 should only have Gaussian at (30, 30)
        assert result[0, 30, 30] == pytest.approx(1.0, abs=1e-5)
        assert result[0, 90, 90] < 1e-5
        # Type 1 should only have Gaussian at (90, 90)
        assert result[1, 90, 90] == pytest.approx(1.0, abs=1e-5)
        assert result[1, 30, 30] < 1e-5

    def test_values_in_range(self):
        """All output values should be in [0, 1]."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        # Create a dense set of junctions
        for y in range(10, 120, 5):
            for x in range(10, 120, 5):
                jmap[0, y, x] = 1.0
        result = apply_gaussian_heatmap(jmap, sigma=2.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_snap_to_one_with_offset(self):
        """With sub-pixel offsets, nearest pixel should be snapped to 1.0.

        This is important for focal loss which requires at least one pixel
        per junction to have target >= threshold for proper normalization.
        """
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        jmap[0, 64, 64] = 1.0
        joff = np.zeros((1, 2, 128, 128), dtype=np.float32)
        joff[0, 0, 64, 64] = 0.3  # dy: center at 64.8
        joff[0, 1, 64, 64] = 0.4  # dx: center at 64.9
        result = apply_gaussian_heatmap(jmap, joff=joff, sigma=1.5)
        # The nearest pixel to (64.8, 64.9) is (65, 65), which should be snapped to 1.0
        assert result[0, 65, 65] == pytest.approx(1.0, abs=1e-6)

    def test_snap_ensures_positive_exists(self):
        """Every junction should have at least one pixel at exactly 1.0."""
        jmap = np.zeros((1, 128, 128), dtype=np.float32)
        jmap[0, 30, 40] = 1.0
        jmap[0, 80, 90] = 1.0
        joff = np.zeros((1, 2, 128, 128), dtype=np.float32)
        # Large offsets
        joff[0, 0, 30, 40] = -0.4
        joff[0, 1, 30, 40] = 0.3
        joff[0, 0, 80, 90] = 0.2
        joff[0, 1, 80, 90] = -0.3
        result = apply_gaussian_heatmap(jmap, joff=joff, sigma=1.5)
        # At least 2 pixels should have value == 1.0 (one per junction)
        assert (result[0] >= 0.999).sum() >= 2


class TestApplyGaussianHeatmapTorch:
    """Tests for the torch version of apply_gaussian_heatmap."""

    def test_basic_functionality(self):
        """Should produce same results as numpy version but as tensor."""
        jmap_np = np.zeros((1, 128, 128), dtype=np.float32)
        jmap_np[0, 64, 64] = 1.0
        jmap_torch = torch.from_numpy(jmap_np)

        result_np = apply_gaussian_heatmap(jmap_np, sigma=2.0)
        result_torch = apply_gaussian_heatmap_torch(jmap_torch, sigma=2.0)

        assert isinstance(result_torch, torch.Tensor)
        assert result_torch.dtype == torch.float32
        np.testing.assert_allclose(
            result_torch.numpy(), result_np, atol=1e-6
        )

    def test_with_offset(self):
        """Torch version should handle offset tensors."""
        jmap = torch.zeros(1, 128, 128)
        jmap[0, 64, 64] = 1.0
        joff = torch.zeros(1, 2, 128, 128)
        joff[0, 0, 64, 64] = 0.1
        joff[0, 1, 64, 64] = -0.2
        result = apply_gaussian_heatmap_torch(jmap, joff=joff, sigma=2.0)
        assert isinstance(result, torch.Tensor)
        assert result.max().item() > 0.9


class TestGaussianFocalLoss:
    """Tests for the gaussian_focal_loss function."""

    def test_import(self):
        """Should be importable from the multitask_learner module."""
        from lcnn.models.multitask_learner import gaussian_focal_loss
        assert callable(gaussian_focal_loss)

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction should yield near-zero loss."""
        from lcnn.models.multitask_learner import gaussian_focal_loss

        # Create a target with one junction center
        target = torch.zeros(2, 128, 128)
        target[0, 64, 64] = 1.0

        # Perfect prediction: logits very high for class 1 at the junction
        logits = torch.zeros(2, 2, 128, 128)
        logits[1, :, :, :] = -10.0  # Strong negative for class 1 (no junction)
        logits[1, 0, 64, 64] = 10.0  # Strong positive for class 1 at junction

        loss = gaussian_focal_loss(logits, target)
        assert loss.shape == (2,)
        assert loss[0].item() < 0.1

    def test_loss_is_positive(self):
        """Loss should always be non-negative."""
        from lcnn.models.multitask_learner import gaussian_focal_loss

        target = torch.zeros(2, 128, 128)
        target[0, 64, 64] = 1.0
        logits = torch.randn(2, 2, 128, 128)

        loss = gaussian_focal_loss(logits, target)
        assert (loss >= 0).all()

    def test_gaussian_target(self):
        """Should work with Gaussian (soft) targets."""
        from lcnn.models.multitask_learner import gaussian_focal_loss

        # Create a Gaussian target
        target = torch.zeros(1, 128, 128)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                dist2 = dy ** 2 + dx ** 2
                target[0, 64 + dy, 64 + dx] = np.exp(-dist2 / (2 * 2.0 ** 2))
        target[0, 64, 64] = 1.0  # Exact center

        logits = torch.randn(2, 1, 128, 128)
        loss = gaussian_focal_loss(logits, target)
        assert loss.shape == (1,)
        assert (loss >= 0).all()

    def test_batch_dimension(self):
        """Loss should return per-sample values for batched input."""
        from lcnn.models.multitask_learner import gaussian_focal_loss

        batch_size = 4
        target = torch.zeros(batch_size, 128, 128)
        for b in range(batch_size):
            target[b, 32 * b + 10, 64] = 1.0

        logits = torch.randn(2, batch_size, 128, 128)
        loss = gaussian_focal_loss(logits, target)
        assert loss.shape == (batch_size,)

    def test_pos_thresh_with_subpixel(self):
        """Focal loss should handle near-1.0 targets with pos_thresh.

        When Gaussians have sub-pixel centers, the peak pixel may not
        be exactly 1.0. The pos_thresh parameter should still identify
        these as positives.
        """
        from lcnn.models.multitask_learner import gaussian_focal_loss

        # Target with near-1.0 value (simulating sub-pixel Gaussian)
        target = torch.zeros(1, 128, 128)
        target[0, 64, 64] = 0.995  # Near but not exactly 1.0

        logits = torch.randn(2, 1, 128, 128)
        loss = gaussian_focal_loss(logits, target, pos_thresh=0.99)
        assert loss.shape == (1,)
        assert (loss >= 0).all()
        # With default pos_thresh=0.99, 0.995 should be counted as positive
        # This should produce a reasonable loss value (not NaN or inf)
        assert torch.isfinite(loss).all()
