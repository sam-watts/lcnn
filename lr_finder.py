#!/usr/bin/env python3
"""LR Range Test for L-CNN

Sweeps the learning rate exponentially from --min-lr to --max-lr over
--num-steps training iterations, recording loss at each step.  Outputs a
plot (lr_finder.png) and CSV (lr_finder.csv) so you can pick a good
base LR for your dataset.
"""

import math
import os
import os.path as osp
import random

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.utils import recursive_to


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("-d", "--devices", default="0", help="Comma separated GPU devices.")
@click.option("--min-lr", default=1e-7, type=float, help="Start of LR sweep.")
@click.option("--max-lr", default=1e-1, type=float, help="End of LR sweep.")
@click.option("--num-steps", default=300, type=int, help="Number of sweep steps.")
@click.option("--output", default="lr_finder_results", type=click.Path(),
              help="Output directory.")
def main(config, devices, min_lr, max_lr, num_steps, output):
    """Run an LR range test using the given CONFIG yaml file."""
    C.update(C.from_yaml(filename=config))
    M.update(C.model)

    outdir = output
    os.makedirs(outdir, exist_ok=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Device
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
    elif torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device_name)
    print(f"Device: {device_name}")

    # DataLoader
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(split="train", data_sources=M.data_sources),
        shuffle=True,
        batch_size=M.batch_size,
        **kwargs,
    )

    # Model
    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model = model.to(device)
    model.train()

    # Optimizer -- start at min_lr
    optim = torch.optim.Adam(
        model.parameters(),
        lr=min_lr,
        weight_decay=C.optim.weight_decay,
        amsgrad=C.optim.amsgrad,
    )

    # Exponential LR multiplier per step
    gamma = (max_lr / min_lr) ** (1.0 / num_steps)

    # Save initial model state so we don't need to recreate
    init_state = {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "optim": {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in optim.state_dict().items()},
    }

    # --- Sweep ---
    lrs = []
    raw_losses = []
    smooth_losses = []
    best_loss = float("inf")
    avg_loss = 0.0
    beta = 0.98  # smoothing factor

    data_iter = iter(train_loader)
    num_stacks = C.model.num_stacks

    print(f"Running LR sweep: {min_lr:.1e} -> {max_lr:.1e} over {num_steps} steps")
    for step in range(num_steps):
        # Get batch (loop back to start if dataset is smaller than num_steps)
        try:
            image, meta, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            image, meta, target = next(data_iter)

        input_dict = {
            "image": recursive_to(image, device),
            "meta": recursive_to(meta, device),
            "target": recursive_to(target, device),
            "mode": "training",
        }

        optim.zero_grad()
        result = model(input_dict)

        # Compute total loss (same as trainer._loss)
        losses = result["losses"]
        total_loss = 0
        for i in range(num_stacks):
            for name, val in losses[i].items():
                total_loss += val.mean()

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  Step {step}: loss is NaN/Inf at lr={optim.param_groups[0]['lr']:.2e}, stopping early")
            break

        total_loss.backward()
        optim.step()

        # Record
        loss_val = total_loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed = avg_loss / (1 - beta ** (step + 1))  # bias-corrected

        current_lr = optim.param_groups[0]["lr"]
        lrs.append(current_lr)
        raw_losses.append(loss_val)
        smooth_losses.append(smoothed)

        if smoothed < best_loss:
            best_loss = smoothed

        # Stop if loss diverges (4x the best smoothed loss)
        if step > 10 and smoothed > 4 * best_loss:
            print(f"  Step {step}: loss diverging at lr={current_lr:.2e}, stopping early")
            break

        if step % 20 == 0:
            print(f"  Step {step:4d}/{num_steps}  lr={current_lr:.2e}  loss={smoothed:.4f}")

        # Increase LR for next step
        for pg in optim.param_groups:
            pg["lr"] *= gamma

    # --- Save results ---
    csv_path = osp.join(outdir, "lr_finder.csv")
    with open(csv_path, "w") as f:
        f.write("lr,raw_loss,smooth_loss\n")
        for lr, rl, sl in zip(lrs, raw_losses, smooth_losses):
            f.write(f"{lr:.8e},{rl:.6f},{sl:.6f}\n")
    print(f"Saved CSV: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, smooth_losses, label="Smoothed loss", linewidth=2)
    ax.plot(lrs, raw_losses, alpha=0.3, label="Raw loss", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title("LR Range Test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark the LR with steepest descent
    if len(smooth_losses) > 10:
        gradients = np.gradient(smooth_losses, np.log10(lrs))
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]
        ax.axvline(x=suggested_lr, color="red", linestyle="--", alpha=0.7,
                    label=f"Suggested LR: {suggested_lr:.2e}")
        ax.legend()
        print(f"\nSuggested LR (steepest descent): {suggested_lr:.2e}")
        print(f"  Typical range to try: [{suggested_lr / 3:.2e}, {suggested_lr * 3:.2e}]")

    plot_path = osp.join(outdir, "lr_finder.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
