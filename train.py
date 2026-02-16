#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
"""

import datetime
import glob
import math
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import torch
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def create_scheduler(optimizer, config, last_epoch=-1):
    """Create LR scheduler based on config.

    Supports:
      - "cosine_warmup": linear warmup for `warmup_epochs`, then cosine decay
        to `min_lr` over the remaining epochs.
      - "plateau": ReduceLROnPlateau -- adapts LR based on validation loss.
        Good when optimal training length is unknown.  Pair with early stopping.
      - "step": original L-CNN behaviour -- divide LR by 10 at `lr_decay_epoch`.
    """
    scheduler_name = config.optim.get("scheduler", "step")

    if scheduler_name == "cosine_warmup":
        warmup_epochs = config.optim.get("warmup_epochs", 3)
        max_epochs = config.optim.max_epoch
        min_lr = config.optim.get("min_lr", 1e-6)
        base_lr = config.optim.lr
        min_lr_ratio = min_lr / base_lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup from min_lr to base_lr
                return min_lr_ratio + (1 - min_lr_ratio) * epoch / max(1, warmup_epochs - 1)
            # Cosine annealing from base_lr to min_lr
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=last_epoch
        )

    elif scheduler_name == "plateau":
        # ReduceLROnPlateau -- no last_epoch; state is restored via
        # load_state_dict when resuming (see main()).
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.optim.get("plateau_factor", 0.5),
            patience=config.optim.get("plateau_patience", 5),
            min_lr=config.optim.get("min_lr", 1e-6),
        )

    elif scheduler_name == "step":
        # Reproduce the original single-step decay
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[config.optim.lr_decay_epoch],
            gamma=0.1,
            last_epoch=last_epoch,
        )

    else:
        # No scheduler -- manual / no decay
        return None


def _build_sampler(dataset, model_cfg):
    """Build a WeightedRandomSampler for balanced dataset training.

    Supports several strategies via model_cfg.sampling_strategy:
      - "uniform"  : no sampler (plain shuffle) -- default
      - "balanced" : every dataset equally likely per sample
      - "sqrt"     : P(dataset_i) ∝ √count_i -- moderate upweighting of small
                     datasets (halfway between uniform and balanced on log scale)
      - list[float]: multiplicative factors on the natural (count-proportional)
                     rate, e.g. [3.0, 1.0, 1.0] samples the first dataset at
                     3× its natural frequency.

    Implementation note: the per-sample weight loop below divides each
    dataset_weight by the dataset's sample count, so a sample in dataset i
    is drawn with probability ∝ dataset_weights[i] / count_i.  The total
    draw probability for dataset i is therefore ∝ dataset_weights[i].
    So dataset_weights must represent the *desired dataset-level sampling
    probabilities* (before normalisation).
    """
    strategy = getattr(model_cfg, "sampling_strategy", "uniform")
    if strategy == "uniform":
        return None

    source_indices = np.array(dataset.source_indices)
    n_sources = len(dataset.source_names)
    counts = np.array([dataset.source_counts[s] for s in dataset.source_names])

    if strategy == "balanced":
        # Equal probability for every dataset
        dataset_weights = np.ones(n_sources) / n_sources
    elif strategy == "sqrt":
        # P(dataset_i) ∝ √count_i -- between uniform (∝ count) and balanced (equal)
        sqrt_counts = np.sqrt(counts)
        dataset_weights = sqrt_counts / sqrt_counts.sum()
    elif isinstance(strategy, (list, tuple)):
        if len(strategy) != n_sources:
            raise ValueError(
                f"sampling_strategy list has {len(strategy)} entries "
                f"but there are {n_sources} data_sources"
            )
        # Multiplicative factors on natural (count-proportional) rate
        factors = np.array(strategy, dtype=float)
        dataset_weights = factors * counts
        dataset_weights /= dataset_weights.sum()
    else:
        raise ValueError(f"Unknown sampling_strategy: {strategy}")

    # Assign per-sample weight based on its dataset
    per_sample_weight = np.zeros(len(dataset))
    for i, dw in enumerate(dataset_weights):
        mask = source_indices == i
        n = mask.sum()
        if n > 0:
            per_sample_weight[mask] = dw / n

    # Print effective epoch composition
    probs = per_sample_weight / per_sample_weight.sum()
    print(f"Sampling strategy: {strategy}")
    for i, name in enumerate(dataset.source_names):
        source_prob = probs[source_indices == i].sum() * 100
        print(f"  {name}: {counts[i]} samples, ~{source_prob:.1f}% of each epoch")

    return torch.utils.data.WeightedRandomSampler(
        weights=per_sample_weight,
        num_samples=len(dataset),
        replacement=True,
    )


def get_outdir():
    # load config
    name = f"{C.io.run_name}-{str(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))}"
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    elif torch.backends.mps.is_available():
        device_name = "mps"
        print("Let's use the Apple Silicon GPU!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset

    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(datadir, split="train")[0]
    # sys.exit(0)

    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    train_dataset = WireframeDataset(split="train", data_sources=M.data_sources)
    sampler = _build_sampler(train_dataset, M)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=M.batch_size,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(split="val", data_sources=M.data_sources),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )
    epoch_size = len(train_loader)

    # Always validate on gfrid_roof_centered in addition to the main dataset
    extra_val_loaders = {}
    if set(M.data_sources) != {"gfrid_roof_centered"}:
        extra_val_loaders["gfrid_roof_centered"] = torch.utils.data.DataLoader(
            WireframeDataset(split="val", data_sources=["gfrid_roof_centered"]),
            shuffle=False,
            batch_size=M.batch_size_eval,
            **kwargs,
        )

    if resume_from:
        ckpt_name = "checkpoint_best.pth" if C.io.get("resume_from_best") else "checkpoint_latest.pth"
        ckpt_path = osp.join(resume_from, ckpt_name)
        print(f"Resuming from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

    # 2. model
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

    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])

    # Override LR if requested (useful for warm-restart from a checkpoint
    # that had already decayed its LR).
    if resume_from and C.optim.get("reset_lr_on_resume"):
        new_lr = C.optim.lr
        for pg in optim.param_groups:
            pg["lr"] = new_lr
        print(f"Reset LR to {new_lr:.2e} (reset_lr_on_resume=true)")

    # 4. lr scheduler
    if resume_from:
        resume_epoch = checkpoint["iteration"] // epoch_size
        if C.optim.get("reset_lr_on_resume"):
            # Fresh scheduler -- don't restore old state
            scheduler = create_scheduler(optim, C)
            print(f"Fresh scheduler from epoch {resume_epoch}, "
                  f"lr = {optim.param_groups[0]['lr']:.2e}")
        else:
            scheduler = create_scheduler(optim, C, last_epoch=resume_epoch - 1)
            # Restore scheduler internal state (essential for ReduceLROnPlateau
            # which cannot reconstruct its state from last_epoch alone).
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Resumed scheduler at epoch {resume_epoch}, "
                  f"lr = {optim.param_groups[0]['lr']:.2e}")
    else:
        scheduler = create_scheduler(optim, C)
    if scheduler is not None:
        print(f"Using scheduler: {C.optim.get('scheduler', 'step')}")
    # When doing a warm restart (reset_lr_on_resume), create a fresh output
    # directory so the new run gets its own logs, checkpoints, and wandb run.
    if resume_from and C.optim.get("reset_lr_on_resume"):
        outdir = get_outdir()
    else:
        outdir = resume_from or get_outdir()
    print("outdir:", outdir)

    try:
        trainer = lcnn.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
            scheduler=scheduler,
            extra_val_loaders=extra_val_loaders,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            if "best_verification_loss" in checkpoint:
                trainer.best_verification_loss = checkpoint["best_verification_loss"]
            if "best_watched_loss" in checkpoint:
                trainer._best_watched_loss = checkpoint["best_watched_loss"]
            if "best_loss_epoch" in checkpoint:
                trainer._best_loss_epoch = checkpoint["best_loss_epoch"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise

    # validate on gfrid only at end of training
    trainer.val_loader = torch.utils.data.DataLoader(
        WireframeDataset(split="val", data_sources=["gfrid"]),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )

    trainer.validate(extra_label="gfrid-only")


if __name__ == "__main__":
    main()
