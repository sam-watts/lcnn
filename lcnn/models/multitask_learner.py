from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict["image"]
        outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]

        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        offset = self.head_off
        loss_weight = M.loss_weight
        # Determine junction heatmap loss function once (not per-stack)
        use_focal_loss = (
            getattr(M, "jmap_loss", "cross_entropy") == "focal"
            and getattr(M, "jmap_gaussian_sigma", 0) > 0
        )
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0] : offset[1]].squeeze(0)
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            if use_focal_loss:
                L["jmap"] = sum(
                    gaussian_focal_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
                )
            else:
                L["jmap"] = sum(
                    cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
                )
            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
                .mean(2)
                .mean(1)
            )
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def gaussian_focal_loss(logits, target, alpha=2.0, beta=4.0, pos_thresh=0.99):
    """Focal loss for Gaussian heatmap regression (CenterNet-style).

    This is the penalty-reduced pixel-wise logistic regression with focal
    loss, as described in "Objects as Points" (Zhou et al., 2019) and
    "CornerNet" (Law & Deng, 2018). It is specifically designed for
    Gaussian heatmap targets where:

    - Pixels at object centers have target = 1.0 (or near 1.0 with sub-pixel offsets)
    - Pixels near centers have 0 < target < 1.0 (from the Gaussian)
    - Background pixels have target = 0.0

    The loss reduces penalties for pixels near (but not exactly on) the
    center, controlled by the beta parameter.

    Note: A threshold (pos_thresh) is used instead of exact equality to
    identify positive pixels. This is necessary because when Gaussians
    are centered at sub-pixel positions, no pixel may have a target value
    of exactly 1.0.

    Args:
        logits: Predicted logits of shape [2, batch, H, W] (2 classes).
        target: Ground truth Gaussian heatmap of shape [batch, H, W],
                with values in [0, 1].
        alpha: Focal loss exponent for hard example mining (default: 2.0).
        beta: Exponent for penalty reduction near Gaussian centers (default: 4.0).
        pos_thresh: Threshold for considering a pixel as a positive (junction
                    center). Pixels with target >= pos_thresh are treated as
                    positives. Default: 0.99.

    Returns:
        Per-sample loss tensor of shape [batch].
    """
    # Get predicted probability for the positive (junction) class
    pred = F.softmax(logits, dim=0)[1]  # [batch, H, W]
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)

    # Separate positive (center) and negative (non-center) pixels.
    # Use a threshold instead of exact equality to handle sub-pixel
    # Gaussian centers where no pixel has target == 1.0 exactly.
    pos_mask = target.ge(pos_thresh).float()
    neg_mask = target.lt(pos_thresh).float()

    # Positive loss: -(1 - p)^alpha * log(p) at junction centers
    pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask

    # Negative loss: -(1 - Y)^beta * p^alpha * log(1 - p) elsewhere
    # The (1 - Y)^beta term reduces penalty for pixels near the Gaussian center
    neg_loss = (
        -torch.pow(1 - target, beta)
        * torch.pow(pred, alpha)
        * torch.log(1 - pred)
        * neg_mask
    )

    loss = pos_loss + neg_loss

    # Normalize by the number of positive pixels (junction centers)
    num_pos = pos_mask.sum(dim=(1, 2)).clamp(min=1)
    return loss.sum(dim=(1, 2)) / num_pos


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
