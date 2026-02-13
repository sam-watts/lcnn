#!/usr/bin/env python3
"""Evaluate a trained L-CNN model on one or more val datasets and log metrics to the original wandb run."""

import os
import os.path as osp
from pathlib import Path

import click
import numpy as np
import torch

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.metric import msTPFP, ap, mAPJ, post_jheatmap
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.utils import recursive_to


def evaluate_dataset(model, dataset_name, device, checkpoint_name):
    """Evaluate the model on a single validation dataset.

    Returns a dict of metric names to values, e.g.
      {"eval-gfrid-checkpoint_best.pth/sAP5": 42.1, ...}
    """
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    val_dataset = WireframeDataset(split="val", data_sources=[dataset_name])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )

    print(f"\nEvaluating on {len(val_dataset)} {dataset_name} validation samples...")

    per_sample = []
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
                pred_lines = H["lines"][i].cpu().numpy()[:, :, :2]
                pred_score = H["score"][i].cpu().numpy()

                for j in range(len(pred_lines)):
                    if j > 0 and (pred_lines[j] == pred_lines[0]).all():
                        pred_lines = pred_lines[:j]
                        pred_score = pred_score[:j]
                        break

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

    # Print results
    print(f"\n=== {dataset_name} Evaluation Results ===")
    print(f"  sAP5:  {sap_results[5]:.1f}")
    print(f"  sAP10: {sap_results[10]:.1f}")
    print(f"  sAP15: {sap_results[15]:.1f}")

    prefix = f"eval-{dataset_name}-{checkpoint_name}"
    return {
        f"{prefix}/sAP5": sap_results[5],
        f"{prefix}/sAP10": sap_results[10],
        f"{prefix}/sAP15": sap_results[15],
    }


def process_log_dir(log_dir, checkpoint_name, val_datasets_arg, device):
    """Load config/model for a single log dir, evaluate, and log to wandb."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {log_dir}")
    print(f"{'=' * 60}")

    # Load config from the run's saved config
    config_file = osp.join(log_dir, "config.yaml")
    if not osp.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)

    # Determine which datasets to evaluate
    if val_datasets_arg:
        val_datasets = [s.strip() for s in val_datasets_arg.split(",")]
    else:
        val_datasets = list(M.data_sources)
    print(f"Val datasets: {val_datasets}")

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

    # Evaluate each dataset and collect all metrics
    all_eval_metrics = {}
    for dataset_name in val_datasets:
        metrics = evaluate_dataset(model, dataset_name, device, checkpoint_name)
        all_eval_metrics.update(metrics)

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

    # Log as summary metrics so they appear in the wandb runs table
    for key, value in all_eval_metrics.items():
        wandb_run.summary[key] = value

    wandb.finish()
    print(f"\nLogged metrics to wandb run: {run_name}")

    # Free GPU memory before next log dir
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()


@click.command()
@click.argument("log_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-d", "--devices", default="0", help="Comma separated GPU devices.")
@click.option("--checkpoint", default="checkpoint_best.pth", help="Checkpoint file to load.")
@click.option(
    "--val-datasets",
    default=None,
    help="Comma separated val dataset names (e.g. gfrid,rid2,roofmapnet). "
         "Defaults to all data_sources in the run config.",
)
def main(log_dirs, devices, checkpoint, val_datasets):
    """Evaluate a trained L-CNN model on one or more val datasets and log metrics
    to the original wandb run.

    LOG_DIRS: One or more paths to log directories (e.g. logs/run-a logs/run-b).
    """
    # Device setup
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
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

    for log_dir in log_dirs:
        process_log_dir(log_dir, checkpoint, val_datasets, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
