"""Tests for Gaussian heatmap generation utilities."""

import numpy as np
import pytest

from lcnn.utils import gaussian_2d, draw_gaussian, generate_gaussian_jmap


class TestGaussian2D:
    def test_shape(self):
        g = gaussian_2d((7, 7), sigma=1.0)
        assert g.shape == (7, 7)

    def test_center_is_one(self):
        g = gaussian_2d((7, 7), sigma=1.0)
        assert g[3, 3] == pytest.approx(1.0)

    def test_symmetric(self):
        g = gaussian_2d((7, 7), sigma=1.0)
        assert g[2, 3] == pytest.approx(g[4, 3])
        assert g[3, 2] == pytest.approx(g[3, 4])
        assert g[2, 2] == pytest.approx(g[4, 4])

    def test_values_decrease_from_center(self):
        g = gaussian_2d((7, 7), sigma=1.0)
        assert g[3, 3] > g[2, 3] > g[1, 3] > g[0, 3]

    def test_different_sigma(self):
        g1 = gaussian_2d((11, 11), sigma=1.0)
        g2 = gaussian_2d((11, 11), sigma=2.0)
        # Larger sigma means wider distribution, so the value at (3, 5)
        # relative to center should be higher for larger sigma
        assert g2[3, 5] > g1[3, 5]

    def test_non_square(self):
        g = gaussian_2d((5, 9), sigma=1.5)
        assert g.shape == (5, 9)
        assert g[2, 4] == pytest.approx(1.0)


class TestDrawGaussian:
    def test_center_value(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (64, 64), sigma=1.0)
        assert heatmap[64, 64] == pytest.approx(1.0)

    def test_in_place_modification(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        result = draw_gaussian(heatmap, (64, 64), sigma=1.0)
        assert result is heatmap

    def test_corner_top_left(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (0, 0), sigma=1.0)
        assert heatmap[0, 0] == pytest.approx(1.0)
        # Should not crash or go out of bounds

    def test_corner_bottom_right(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (127, 127), sigma=1.0)
        assert heatmap[127, 127] == pytest.approx(1.0)

    def test_max_not_sum(self):
        """Overlapping Gaussians should use max, not sum."""
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (64, 64), sigma=2.0)
        draw_gaussian(heatmap, (65, 64), sigma=2.0)
        # Center values should not exceed 1.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_values_in_range(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (64, 64), sigma=2.0)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_out_of_bounds_ignored(self):
        """Drawing at a position outside the heatmap should not crash."""
        heatmap = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(heatmap, (-5, 64), sigma=1.0)
        draw_gaussian(heatmap, (200, 64), sigma=1.0)
        draw_gaussian(heatmap, (64, -5), sigma=1.0)
        draw_gaussian(heatmap, (64, 200), sigma=1.0)
        # Should not have drawn anything meaningful
        assert heatmap.sum() == pytest.approx(0.0, abs=1e-6)


class TestGenerateGaussianJmap:
    def test_single_junction(self):
        jmap = generate_gaussian_jmap((128, 128), [(50, 60)], sigma=1.0)
        assert jmap.shape == (128, 128)
        assert jmap[50, 60] == pytest.approx(1.0)

    def test_multiple_junctions(self):
        junctions = [(10, 20), (50, 60), (100, 110)]
        jmap = generate_gaussian_jmap((128, 128), junctions, sigma=1.0)
        for r, c in junctions:
            assert jmap[r, c] == pytest.approx(1.0)

    def test_empty_junctions(self):
        jmap = generate_gaussian_jmap((128, 128), [], sigma=1.0)
        assert jmap.sum() == 0.0

    def test_output_dtype(self):
        jmap = generate_gaussian_jmap((128, 128), [(50, 50)], sigma=1.0)
        assert jmap.dtype == np.float32

    def test_different_sigmas_produce_different_spread(self):
        jmap_small = generate_gaussian_jmap((128, 128), [(64, 64)], sigma=0.5)
        jmap_large = generate_gaussian_jmap((128, 128), [(64, 64)], sigma=2.0)
        # Larger sigma should have more non-zero pixels
        assert (jmap_large > 0.01).sum() > (jmap_small > 0.01).sum()

    def test_values_bounded(self):
        junctions = [(i * 10, j * 10) for i in range(1, 12) for j in range(1, 12)]
        jmap = generate_gaussian_jmap((128, 128), junctions, sigma=2.0)
        assert jmap.min() >= 0.0
        assert jmap.max() <= 1.0 + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
