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
                return min_lr_ratio + (1 - min_lr_ratio) * epoch / max(1, warmup_epochs)
            # Cosine annealing from base_lr to min_lr
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=last_epoch
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
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(split="train", data_sources=M.data_sources),
        shuffle=True,
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
    # print("epoch_size (train):", epoch_size)
    # print("epoch_size (valid):", len(val_loader))

    if resume_from:
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth"))

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

    # 4. lr scheduler
    if resume_from:
        resume_epoch = checkpoint["iteration"] // epoch_size
        scheduler = create_scheduler(optim, C, last_epoch=resume_epoch - 1)
        print(f"Resumed scheduler at epoch {resume_epoch}, "
              f"lr = {optim.param_groups[0]['lr']:.2e}")
    else:
        scheduler = create_scheduler(optim, C)
    if scheduler is not None:
        print(f"Using scheduler: {C.optim.get('scheduler', 'step')}")
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
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            if "best_verification_loss" in checkpoint:
                trainer.best_verification_loss = checkpoint["best_verification_loss"]
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
