# Code Review: Gaussian Heatmap Implementation

## Summary
This PR replaces binary junction heatmaps with 2D Gaussian distributions for junction detection training in L-CNN. The implementation follows the CenterNet/CornerNet approach from keypoint detection literature.

---

## Review of Changes

### `lcnn/utils.py` - Gaussian Heatmap Utilities

**Strengths:**
- Efficient bounding-box computation (only computes within 3-sigma radius) avoids O(H*W) computation per junction
- Clean separation between the low-level `gaussian_2d` and high-level `apply_gaussian_heatmap`
- Element-wise maximum (not sum) for overlapping Gaussians keeps values in [0, 1]
- Sub-pixel offset support for precise Gaussian center placement

**Issue found and fixed:** When using sub-pixel offsets from `joff`, the Gaussian peak may not land on any integer pixel. This means no pixel would have target == 1.0, which breaks focal loss normalization. **Fixed** by snapping the nearest integer pixel to 1.0 for each junction.

### `lcnn/models/multitask_learner.py` - Focal Loss

**Strengths:**
- Penalty-reduced focal loss is the standard approach for Gaussian heatmap targets
- Well-documented with reference to CenterNet/CornerNet papers
- Configurable alpha/beta hyperparameters

**Issue found and fixed:** The `pos_mask = target.eq(1)` check would fail with sub-pixel Gaussians where no pixel has exactly target == 1.0. **Fixed** by using a threshold (`pos_thresh=0.99`) instead of exact equality.

**Minor issue found and fixed:** Config lookups (`getattr(M, "jmap_loss", ...)`) were inside the per-stack loop but are invariant. Moved outside the loop.

### `lcnn/datasets.py` - Runtime Gaussian Application

**Strengths:**
- Runtime application avoids dataset re-processing
- Uses `getattr` with default for backward compatibility
- Gaussian parameters cached in `__init__` for multiprocessing workers

**Note:** The `sigmoid_l1_loss` for `joff` uses `T["jmap"]` as a weighting mask. With Gaussian targets, this means offset supervision is weighted by Gaussian proximity to junction centers - actually desirable behavior (stronger supervision near centers, weaker at periphery).

### `config/wireframe.yaml` - Configuration

- `jmap_gaussian_sigma: 1.5` is a reasonable default (covers ~9 pixel radius at 128x128)
- `jmap_loss: cross_entropy` as default maintains backward compatibility
- Well-documented with comments explaining valid ranges and options

### Tests

- 28 tests covering core functions, edge cases, and integration
- Tests for the snap-to-one behavior and threshold-based focal loss
- Good coverage of boundary conditions (corners, edges, out-of-bounds)

## Recommendations for Future Work

1. **Consider adaptive sigma**: For datasets with varying image scales or junction densities, an adaptive sigma (e.g., proportional to nearest junction distance) could improve results.

2. **Benchmark data loading overhead**: The per-junction Gaussian computation adds to data loading time. For images with many junctions (100+), consider profiling. The bounding-box optimization should keep this fast.

3. **Experiment with focal loss hyperparameters**: The defaults (alpha=2, beta=4) come from CenterNet. For junction detection on wireframe data, these may benefit from tuning.

## Verdict
**Approve with the fixes applied.** The implementation is clean, well-tested, backward-compatible, and follows established best practices from the keypoint detection literature. The review-identified issues (sub-pixel threshold, pixel snapping, config caching) have been addressed in the final commit.
