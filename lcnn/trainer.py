import atexit
import os
import os.path as osp
import shutil
import threading
import time
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from skimage import io

from lcnn.config import C, M
from lcnn.metric import msTPFP, ap
from lcnn.utils import recursive_to


class Trainer(object):
    def __init__(self, device, model, optimizer, train_loader, val_loader, out,
                 scheduler=None, extra_val_loaders=None):
        self.device = device

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.extra_val_loaders = extra_val_loaders or {}
        self.batch_size = C.model.batch_size

        self.validation_interval = C.io.get("validation_interval", 1)

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.run_wandb()
        time.sleep(1)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000
        self.best_verification_loss = 1e1000

        self.early_stopping_patience = C.optim.get("early_stopping_patience", 0)
        self._best_loss_epoch = 0
        self._stop_training = False

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def run_wandb(self):
        wandb_out = osp.join(self.out, "wandb")
        if not osp.exists(wandb_out):
            os.makedirs(wandb_out)
        self.wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "lcnn"),
            name=C.io.run_name,
            dir=wandb_out,
        )

        def finish():
            if wandb.run is not None:
                wandb.finish()

        atexit.register(finish)

    def _loss(self, result):
        losses = result["losses"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self, extra_label=None):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * M.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * M.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }
                result = self.model(input_dict)

                total_loss += self._loss(result)

                H = result["preds"]
                for i in range(H["jmap"].shape[0]):
                    index = batch_idx * M.batch_size_eval + i
                    np.savez(
                        f"{npz}/{index:06}.npz",
                        **{k: v[i].cpu().numpy() for k, v in H.items()},
                    )
                    if index >= 5:
                        continue
                    self._plot_samples(i, index, H, meta, target, f"{viz}/{index:06}")

        label = f"validation{'' if extra_label is None else '-' + extra_label}"
        self._write_metrics(len(self.val_loader), total_loss, label, True)

        self.mean_loss = total_loss / len(self.val_loader)

        # Compute verification loss (lpos + lneg) for checkpointing.
        # These are the losses most directly correlated with SAP metrics,
        # and are less noisy than the total loss (which is dominated by jmap).
        verification_loss = 0
        for name in ["lpos", "lneg"]:
            if name in self.loss_labels:
                j = self.loss_labels.index(name)
                verification_loss += self.metrics[0, j] / len(self.val_loader)
        if self.wandb_run is not None:
            wandb.log(
                {"validation/verification_loss": verification_loss},
                step=self.iteration,
            )

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
                "best_verification_loss": self.best_verification_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if verification_loss < self.best_verification_loss:
            self.best_verification_loss = verification_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        # Early stopping check
        if (self.early_stopping_patience > 0
                and self.epoch - self._best_loss_epoch >= self.early_stopping_patience):
            epochs_since = self.epoch - self._best_loss_epoch
            pprint(
                f"Early stopping: no val loss improvement for "
                f"{epochs_since} epochs (best at epoch {self._best_loss_epoch})"
            )
            self._stop_training = True
            if self.wandb_run is not None:
                wandb.log({"early_stopped_epoch": self.epoch}, step=self.iteration)

        # Run extra validations (e.g. always validate on gfrid_roof_centered)
        for name, loader in self.extra_val_loaders.items():
            self._validate_extra(loader, name)

        if training:
            self.model.train()

    def _collect_sample_preds(self, H, i, dataset, index):
        """Extract predicted/GT lines for a single sample for sAP computation."""
        pred_lines = H["lines"][i].cpu().numpy()[:, :, :2]
        pred_score = H["score"][i].cpu().numpy()

        # Trim padded/duplicate lines (model pads by wrapping to first line)
        for j in range(len(pred_lines)):
            if j > 0 and (pred_lines[j] == pred_lines[0]).all():
                pred_lines = pred_lines[:j]
                pred_score = pred_score[:j]
                break

        with np.load(dataset.filelist[index]["label"]) as npz_file:
            gt_lines = npz_file["lpos"][:, :, :2]

        return {
            "pred_lines": pred_lines,
            "pred_score": pred_score,
            "gt_lines": gt_lines,
        }

    def _compute_sap(self, per_sample):
        """Compute sAP at thresholds 5, 10, 15 from collected predictions."""
        n_gt = sum(len(s["gt_lines"]) for s in per_sample)
        if n_gt == 0:
            return {5: 0.0, 10: 0.0, 15: 0.0}

        results = {}
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
                tp_cum = np.cumsum(tp_cat[idx]) / n_gt
                fp_cum = np.cumsum(fp_cat[idx]) / n_gt
                results[threshold] = 100 * ap(tp_cum, fp_cum)
            else:
                results[threshold] = 0.0
        return results

    def _log_sap(self, sap, prefix):
        """Print and log sAP metrics to wandb."""
        pprint(
            f"  {prefix} sAP5: {sap[5]:.1f}  "
            f"sAP10: {sap[10]:.1f}  sAP15: {sap[15]:.1f}"
        )
        if self.wandb_run is not None:
            wandb.log(
                {
                    f"{prefix}/sAP5": sap[5],
                    f"{prefix}/sAP10": sap[10],
                    f"{prefix}/sAP15": sap[15],
                },
                step=self.iteration,
            )

    def _validate_extra(self, loader, label):
        """Run validation on an additional dataset (no checkpoint save)."""
        tprint(f"Running extra validation ({label})...", " " * 75)

        total_loss = 0
        self.metrics[...] = 0
        per_sample = []

        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }
                result = self.model(input_dict)
                total_loss += self._loss(result)

                H = result["preds"]
                for i in range(H["jmap"].shape[0]):
                    index = batch_idx * M.batch_size_eval + i
                    per_sample.append(
                        self._collect_sample_preds(H, i, loader.dataset, index)
                    )

        prefix = f"validation-{label}"
        self._write_metrics(len(loader), total_loss, prefix, True)

        sap = self._compute_sap(per_sample)
        self._log_sap(sap, prefix)

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (image, meta, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "mode": "training",
            }
            result = self.model(input_dict)

            loss = self._loss(result)
            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self._write_metrics(1, loss.item(), "training", do_print=False)

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        log_data = {}
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                log_data[f"{prefix}/{i}/{label}"] = metric / size
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        log_data[f"{prefix}/total_loss"] = total_loss / size
        log_data["lr"] = self.optim.param_groups[0]["lr"]
        if self.wandb_run is not None:
            wandb.log(log_data, step=self.iteration)
        return total_loss

    def _plot_samples(self, i, index, result, meta, target, prefix):
        fn = self.val_loader.dataset.get_image_path(index)
        img = io.imread(fn)
        img_path = f"{prefix}_img.jpg"
        imshow(img)
        plt.savefig(img_path)
        plt.close()

        mask_result = result["jmap"][i].cpu().numpy()
        mask_target = target["jmap"][i].cpu().numpy()
        mask_paths = []
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            mask_a = f"{prefix}_mask_{ch}a.jpg"
            mask_b = f"{prefix}_mask_{ch}b.jpg"
            imshow(ia)
            plt.savefig(mask_a)
            plt.close()
            imshow(ib)
            plt.savefig(mask_b)
            plt.close()
            mask_paths.extend([mask_a, mask_b])

        line_result = result["lmap"][i].cpu().numpy()
        line_target = target["lmap"][i].cpu().numpy()
        line_target_path = f"{prefix}_line_a.jpg"
        line_result_path = f"{prefix}_line_b.jpg"
        imshow(line_target)
        plt.savefig(line_target_path)
        plt.close()
        imshow(line_result)
        plt.savefig(line_result_path)
        plt.close()

        def draw_vecl(lines, sline, juncs, junts, fn):
            imshow(img)
            if len(lines) > 0:
                # Deduplicate lines (handles both GT duplicates and
                # model-output padding that wraps via modular indexing)
                seen = set()
                for (a, b), s in zip(lines, sline):
                    key = (a[0], a[1], b[0], b[1])
                    if key in seen:
                        continue
                    seen.add(key)
                    plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
            if len(juncs) > 0:
                juncs_unique = np.unique(juncs, axis=0)
                plt.scatter(juncs_unique[:, 1], juncs_unique[:, 0],
                            c="red", s=64, zorder=100)
            if junts is not None and len(junts) > 0:
                junts_unique = np.unique(junts, axis=0)
                plt.scatter(junts_unique[:, 1], junts_unique[:, 0],
                            c="blue", s=64, zorder=100)
            plt.savefig(fn)
            plt.close()

        junc = meta[i]["junc"].cpu().numpy() * 4
        jtyp = meta[i]["jtyp"].cpu().numpy()
        juncs = junc[jtyp == 0]
        junts = junc[jtyp == 1]
        rjuncs = result["juncs"][i].cpu().numpy() * 4
        rjunts = None
        if "junts" in result:
            rjunts = result["junts"][i].cpu().numpy() * 4

        lpre = meta[i]["lpre"].cpu().numpy() * 4
        vecl_target = meta[i]["lpre_label"].cpu().numpy()
        vecl_result = result["lines"][i].cpu().numpy() * 4
        score = result["score"][i].cpu().numpy()
        lpre = lpre[vecl_target == 1]

        vecl_target_path = f"{prefix}_vecl_a.jpg"
        vecl_result_path = f"{prefix}_vecl_b.jpg"
        draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts, vecl_target_path)
        draw_vecl(vecl_result, score, rjuncs, rjunts, vecl_result_path)

        if self.wandb_run is not None:
            base_key = f"validation/sample_{index:06d}"
            log_data = {
                f"{base_key}/image": wandb.Image(img_path),
                f"{base_key}/line_target": wandb.Image(line_target_path),
                f"{base_key}/line_result": wandb.Image(line_result_path),
                f"{base_key}/vecl_target": wandb.Image(vecl_target_path),
                f"{base_key}/vecl_result": wandb.Image(vecl_result_path),
            }
            for mask_path in mask_paths:
                mask_key = mask_path.replace(prefix + "_", "")
                log_data[f"{base_key}/{mask_key}"] = wandb.Image(mask_path)
            wandb.log(log_data, step=self.iteration)

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        # Reset patience counter so resumed runs get a fresh window
        self._best_loss_epoch = start_epoch
        for self.epoch in range(start_epoch, self.max_epoch):
            current_lr = self.optim.param_groups[0]["lr"]
            if self.scheduler is not None:
                pprint(f"Epoch {self.epoch}: lr = {current_lr:.2e}")
            elif self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()
            if (self.epoch + 1) % self.validation_interval == 0:
                self.validate()
            if self._stop_training:
                pprint(f"Training stopped early at epoch {self.epoch}")
                break
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(self.mean_loss)
                else:
                    self.scheduler.step()


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(sm, fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


