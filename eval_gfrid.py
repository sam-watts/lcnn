#!/usr/bin/env python3
"""Evaluate a trained L-CNN model on gfrid and log metrics to the original wandb run.

Usage:
    eval_gfrid.py <log-dir> [options]
    eval_gfrid.py (-h | --help)

Arguments:
    <log-dir>       Path to the logs directory (e.g. logs/260211-221828-5d4d784-rid2_gfrid)

Options:
    -h --help                       Show this screen.
    -d --devices <devices>          Comma separated GPU devices [default: 0]
    --checkpoint <name>             Checkpoint file to load [default: checkpoint_best.pth]
"""

import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.metric import msTPFP, ap, mAPJ, post_jheatmap
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.utils import recursive_to


def main():
    args = docopt(__doc__)
    log_dir = args["<log-dir>"]
    checkpoint_name = args["--checkpoint"]

    # Load config from the run's saved config
    config_file = osp.join(log_dir, "config.yaml")
    if not osp.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)

    run_name = C.io.run_name
    wandb_dir = Path(log_dir) / "wandb/wandb"

    run_dirs = sorted(
        wandb_dir.glob("run-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    latest_run = run_dirs[0]
    run_id = latest_run.name.split("-")[-1]
    print(f"Run name: {run_name}")
    print(f"Log dir: {log_dir}")
    print(f"Wandb dir: {wandb_dir}")
    print(f"Run ID: {run_id}")

    # Device setup
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device_name = "mps"
        print("Using MPS")
    else:
        print("Using CPU")
    device = torch.device(device_name)

    # Load model
    checkpoint_path = osp.join(log_dir, checkpoint_name)
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load gfrid validation dataset
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    val_dataset = WireframeDataset(split="val", data_sources=["gfrid"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )

    # Run inference once and store per-sample predictions + GT
    print(f"Evaluating on {len(val_dataset)} gfrid validation samples...")

    per_sample = []  # list of dicts with pred_lines, pred_score, gt_lines, gt_junc
    all_junc_pred = np.zeros((0, 3))
    all_junc_pred_offset = np.zeros((0, 3))
    all_junc_gt = []
    all_junc_ids = np.zeros(0, dtype=np.int32)

    sample_idx = 0
    with torch.no_grad():
        for batch_idx, (image, meta, target) in enumerate(val_loader):
            input_dict = {
                "image": recursive_to(image, device),
                "meta": recursive_to(meta, device),
                "target": recursive_to(target, device),
                "mode": "validation",
            }
            result = model(input_dict)
            H = result["preds"]

            batch_size = H["jmap"].shape[0]
            for i in range(batch_size):
                # Extract predicted lines and scores
                pred_lines = H["lines"][i].cpu().numpy()[:, :, :2]
                pred_score = H["score"][i].cpu().numpy()

                # Trim padding (repeated first line indicates end of valid predictions)
                for j in range(len(pred_lines)):
                    if j > 0 and (pred_lines[j] == pred_lines[0]).all():
                        pred_lines = pred_lines[:j]
                        pred_score = pred_score[:j]
                        break

                # Load GT from dataset labels
                label_path = val_dataset.filelist[sample_idx]["label"]
                with np.load(label_path) as npz:
                    gt_lines = npz["lpos"][:, :, :2]
                    gt_junc = npz["junc"][:, :2]

                per_sample.append(
                    {
                        "pred_lines": pred_lines,
                        "pred_score": pred_score,
                        "gt_lines": gt_lines,
                    }
                )

                # Junction predictions via heatmap post-processing
                jmap = H["jmap"][i].cpu().numpy()
                joff = H["joff"][i].cpu().numpy()
                jun_c = post_jheatmap(jmap[0])
                jun_o_c = post_jheatmap(jmap[0], offset=joff[0])

                all_junc_pred = np.vstack((all_junc_pred, jun_c))
                all_junc_pred_offset = np.vstack((all_junc_pred_offset, jun_o_c))
                all_junc_gt.append(gt_junc)
                all_junc_ids = np.hstack(
                    (all_junc_ids, np.array([sample_idx] * len(jun_c)))
                )

                sample_idx += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_idx}/{len(val_dataset)} samples")

    print(f"Processed {sample_idx} samples total")

    # Compute sAP at thresholds 5, 10, 15
    n_gt_lines = sum(len(s["gt_lines"]) for s in per_sample)
    sap_results = {}
    for threshold in [5, 10, 15]:
        all_tp, all_fp, all_scores = [], [], []
        for s in per_sample:
            if len(s["pred_lines"]) > 0 and len(s["gt_lines"]) > 0:
                tp, fp = msTPFP(s["pred_lines"], s["gt_lines"], threshold)
                all_tp.append(tp)
                all_fp.append(fp)
                all_scores.append(s["pred_score"])

        if all_tp:
            tp_cat = np.concatenate(all_tp)
            fp_cat = np.concatenate(all_fp)
            scores_cat = np.concatenate(all_scores)
            idx = np.argsort(-scores_cat)
            tp_cum = np.cumsum(tp_cat[idx]) / n_gt_lines
            fp_cum = np.cumsum(fp_cat[idx]) / n_gt_lines
            sap_results[threshold] = 100 * ap(tp_cum, fp_cum)
        else:
            sap_results[threshold] = 0.0

    # Compute mAPJ
    # junc_distances = [0.5, 1.0, 2.0]
    # all_junc_ids = all_junc_ids.astype(np.int64)
    # mapj = mAPJ(all_junc_pred, all_junc_gt, junc_distances, all_junc_ids)
    # mapj_offset = mAPJ(all_junc_pred_offset, all_junc_gt, junc_distances, all_junc_ids)

    # Print results
    print("\n=== gfrid Evaluation Results ===")
    print(f"  sAP5:  {sap_results[5]:.1f}")
    print(f"  sAP10: {sap_results[10]:.1f}")
    print(f"  sAP15: {sap_results[15]:.1f}")
    # print(f"  mAPJ:          {mapj:.1f}")
    # print(f"  mAPJ (offset): {mapj_offset:.1f}")

    # Log to wandb, resuming the original run
    import wandb

    wandb_out = osp.join(log_dir, "wandb")
    os.makedirs(wandb_out, exist_ok=True)

    wandb_run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "lcnn"),
        name=run_name,
        id=run_id,
        resume="must",
        dir=wandb_out,
    )
    eval_metrics = {
        f"eval-gfrid-{checkpoint_name}/sAP5": sap_results[5],
        f"eval-gfrid-{checkpoint_name}/sAP10": sap_results[10],
        f"eval-gfrid-{checkpoint_name}/sAP15": sap_results[15],
        # f"eval-gfrid-{checkpoint_name}/mAPJ": mapj,
        # f"eval-gfrid-{checkpoint_name}/mAPJ_offset": mapj_offset,
    }

    # Log as summary metrics so they appear in the wandb runs table
    for key, value in eval_metrics.items():
        wandb_run.summary[key] = value

    wandb.finish()
    print("\nLogged metrics to wandb run:", run_name)
    print("Done.")


if __name__ == "__main__":
    main()
