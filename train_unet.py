"""Train the U-Net (unet.py) from scratch on py3DCal's DIGIT calibration data.

Labels: MSE to the probe-sphere gradient labels (ground truth from the annotated probe
positions), as py3DCal trains TouchNet.

Distillation (--distill-weight > 0): MSE to the pretrained TouchNet's gradients as well.
Every label is a 2 mm sphere, so a label-only U-Net (with a whole-frame receptive field)
learns to draw spheres and fails on other shapes (data/.../test_objects). TouchNet's output
on those same sphere frames is sphere-shaped too, so --composite-fraction of the distillation
inputs are sums of 2-3 probe-frame differences: multi-contact frames with no label, where
only TouchNet's output is the target (or, with --composite-labels and no distillation, sums of
non-overlapping probes with the summed sphere labels). TouchNet's outputs are computed once into
--teacher-cache (TouchNet is ~20x the U-Net's compute).

Split (--split py3dcal, default): 20% of the probe positions at random (sklearn
train_test_split, random_state=42), as py3DCal's train_model() holds out data. If the
published TouchNet was trained with that default, both models are evaluated on positions
neither has seen. --split last holds out the last 10% of probe positions as
train_fast_touchnet.py does: a strip along the top edge of the gel (y 16.5-18 mm) which the
published TouchNet most likely trained on.

Each epoch reports depth error vs labels and vs TouchNet on held-out probes, and vs TouchNet
on the 3 test objects (no ground truth there). The final epoch is saved.

  python train_unet.py --base 32 --epochs 20                     # labels only
  python train_unet.py --distill-weight 1 --label-weight 0.5     # + distillation and composites
  python train_unet.py --composite-labels                         # labels only, labeled composites
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

import py3DCal as p3d
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings

from touchnet_fp16 import TouchNetFP16
from train_fast_touchnet import ROOT, ProbeImages, depth, load, split_rows
from unet import UNet

TEST_OBJECTS = ("hemispheres", "pawn", "pill")


class Composites(Dataset):
    """(x, label, has_label) training items.

    Item i is probe frame i, or (with probability `fraction`) frame i's difference image plus
    those of 1-2 random other frames: a multi-contact input. Unlabeled by default (distillation
    target only). With `labeled`, partners are drawn whose probe centres are at least
    `min_dist` px from each other, so the sphere labels do not overlap and the summed label is
    the exact gradient of the combined depth map.
    """

    def __init__(self, images, fraction=0.0, seed=0, labeled=False, min_dist=0.0):
        self.images, self.labeled = images, labeled
        n = len(images)
        rng = np.random.default_rng(seed)
        self.recipes = np.full((n, 3), -1, dtype=np.int64)
        self.recipes[:, 0] = np.arange(n)
        mixed = rng.random(n) < fraction
        three = mixed & (rng.random(n) < 0.5)
        if not labeled:
            partners = rng.integers(0, n, (n, 2))
            self.recipes[mixed, 1] = partners[mixed, 0]
            self.recipes[three, 2] = partners[three, 1]
            return
        pos = images.rows[["x_px", "y_px"]].to_numpy(np.float64)
        for i in np.flatnonzero(mixed):
            chosen = [i]
            for _ in range(2 if three[i] else 1):
                for _ in range(50):  # rejection sampling; most frames have far partners
                    j = int(rng.integers(n))
                    if all(np.hypot(*(pos[j] - pos[k])) >= min_dist for k in chosen):
                        chosen.append(j)
                        break
            self.recipes[i, :len(chosen)] = chosen

    def key(self):
        """Identifies the inputs (rows and recipes), to validate a teacher cache."""
        h = hashlib.sha256("\n".join(self.images.rows.img_name).encode())
        h.update(self.recipes.tobytes())
        return h.hexdigest()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        parts = [j for j in self.recipes[i] if j >= 0]
        x, label = self.images[parts[0]]
        if len(parts) == 1:
            return x, label, True
        x = x.clone()
        if self.labeled:
            label = label.clone()
        for j in parts[1:]:
            xj, lj = self.images[j]
            x[:3] += xj[:3]
            if self.labeled:
                label += lj
        return x, (label if self.labeled else torch.zeros_like(label)), self.labeled


class WithTeacher(Dataset):
    """Composites items plus the cached TouchNet gradients of the same input."""

    def __init__(self, items, cache_path):
        self.items, self.cache_path, self.cache = items, cache_path, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.cache is None:  # open in each worker process
            self.cache = np.load(self.cache_path, mmap_mode="r")
        return (*self.items[i], torch.from_numpy(self.cache[i].astype(np.float32)))


@torch.inference_mode()
def build_teacher_cache(teacher, items, path, workers):
    """TouchNet gradients of every training input, fp16 on disk (~8 GB for 25k frames).

    Computed once with TouchNetFP16 (~0.001 px depth from fp32): running TouchNet (~298 GMAC
    per frame) in the training loop would cost ~20x the U-Net itself.
    """
    key_file = path.with_suffix(".key")
    if path.is_file() and key_file.is_file() and key_file.read_text() == items.key():
        print(f"Using cached TouchNet targets {path}", flush=True)
        return
    h, w = items[0][0].shape[-2:]
    fast = TouchNetFP16(teacher.state_dict(), h, w, cuda_graph=False)
    tmp = path.with_suffix(".tmp.npy")
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(items), 2, h, w))
    started, i = time.monotonic(), 0
    for x, _, _ in DataLoader(items, batch_size=16, num_workers=workers):
        diff = x[:, :3].cuda().half().contiguous(memory_format=torch.channels_last)
        cache[i:i + len(x)] = fast.network(diff).cpu().numpy()
        i += len(x)
    cache.flush()
    del cache
    tmp.replace(path)
    key_file.write_text(items.key())
    print(f"Cached TouchNet targets for {i} inputs in {time.monotonic() - started:.0f}s -> {path}", flush=True)


def py3dcal_split(data, val_limit):
    """(train_rows, val_rows, sphere, blank) split by probe position like py3DCal's split_dataset()."""
    train_rows, val_rows, sphere, blank = split_rows(data, 0.1, 10 ** 9)
    rows = pd.concat([train_rows, val_rows])
    ann = pd.read_csv(data / "annotations" / "annotations.csv")
    coords = ann[["x_mm", "y_mm"]].drop_duplicates().reset_index(drop=True)
    _, val_coords = train_test_split(coords, train_size=0.8, random_state=42)
    held_out = pd.MultiIndex.from_frame(rows[["x_mm", "y_mm"]]).isin(pd.MultiIndex.from_frame(val_coords))
    train, val = rows[~held_out], rows[held_out]
    val = val.iloc[np.linspace(0, len(val) - 1, min(val_limit, len(val))).astype(int)]
    return train, val, sphere, blank


@torch.inference_mode()
def evaluate(model, teacher, loader):
    """Mean depth RMSE / peak error (px) of model vs labels and vs TouchNet, and grad MSE vs labels."""
    out = {k: [] for k in ("grad_mse", "label_rmse", "label_peak", "teacher_rmse", "teacher_peak")}
    for x, label in loader:
        x = x.cuda()
        pred = model(x).float()
        ref = teacher(x).float()
        out["grad_mse"].append(F.mse_loss(pred.cpu(), label).item())
        for p, t, l in zip(pred.cpu().numpy(), ref.cpu().numpy(), label.numpy()):
            dp, dt, dl = depth(p), depth(t), depth(l)
            out["label_rmse"].append(np.sqrt(np.mean((dp - dl) ** 2)))
            out["label_peak"].append(abs(dp.max() - dl.max()))
            out["teacher_rmse"].append(np.sqrt(np.mean((dp - dt) ** 2)))
            out["teacher_peak"].append(abs(dp.max() - dt.max()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def object_inputs(data):
    blank = load(data / "blank_images" / "blank.png")
    return {name: add_coordinate_embeddings(torch.from_numpy(np.ascontiguousarray(
                (load(data / "test_objects" / f"{name}.png") - blank).transpose(2, 0, 1))))[None]
            for name in TEST_OBJECTS}


@torch.inference_mode()
def evaluate_objects(model, teacher, inputs):
    """Depth RMSE (px) vs TouchNet on each test object (they have no ground truth)."""
    return {name: float(np.sqrt(np.mean((depth(model(x.cuda())[0].float().cpu().numpy())
                                         - depth(teacher(x.cuda())[0].float().cpu().numpy())) ** 2)))
            for name, x in inputs.items()}


def report(r, objects=None):
    text = (f"vs labels: rmse={r['label_rmse']:.3f}px peak={r['label_peak']:.3f}px grad_mse={r['grad_mse']:.4f} | "
            f"vs TouchNet: rmse={r['teacher_rmse']:.3f}px peak={r['teacher_peak']:.3f}px")
    if objects:
        text += " | objects vs TouchNet: " + " ".join(f"{k}={v:.2f}" for k, v in objects.items())
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "digit_calibration_data")
    parser.add_argument("--base", type=int, default=32, help="U-Net width (channels of the first level)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-weight", type=float, default=1.0, help="Weight of MSE to the sphere labels")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="Weight of MSE to TouchNet's gradients; 0 = labels only")
    parser.add_argument("--composite-fraction", type=float, default=0.5,
                        help="With distillation or --composite-labels: share of inputs that are 2-3 "
                             "probe frames summed")
    parser.add_argument("--composite-labels", action="store_true",
                        help="Composites of non-overlapping probes, labeled with the summed sphere labels")
    parser.add_argument("--composite-min-dist", type=float, default=80.0,
                        help="With --composite-labels: min distance (px) between probe centres; the "
                             "sphere label radius is 36 px")
    parser.add_argument("--split", choices=("py3dcal", "last"), default="py3dcal",
                        help="Held-out probe positions: py3DCal's random 20%% or the last 10%% (see above)")
    parser.add_argument("--val-images", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher-cache", type=Path, default=ROOT / "data" / "touchnet_targets_train.npy",
                        help="fp16 TouchNet gradients of the training inputs (built when missing, ~8 GB)")
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"unet_b{args.base}.pth"
    if args.label_weight <= 0 and args.distill_weight <= 0:
        parser.error("Need a positive --label-weight or --distill-weight")

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    teacher = p3d.models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT,
                                  root=ROOT / "models" / "py3dcal").cuda().eval()
    if args.split == "py3dcal":
        train_rows, val_rows, sphere, blank = py3dcal_split(args.data, args.val_images)
    else:
        train_rows, val_rows, sphere, blank = split_rows(args.data, 0.1, args.val_images)
    print(f"{len(train_rows)} train / {len(val_rows)} val images", flush=True)
    distill = args.distill_weight > 0
    train = Composites(ProbeImages(args.data, train_rows, sphere, blank),
                       args.composite_fraction if distill or args.composite_labels else 0.0,
                       labeled=args.composite_labels, min_dist=args.composite_min_dist)
    if args.composite_labels:
        sizes = (train.recipes >= 0).sum(1)
        print("Composite inputs: " + ", ".join(f"{(sizes == k).sum()} x {k} probe(s)" for k in (1, 2, 3)),
              flush=True)
    if distill:
        build_teacher_cache(teacher, train, args.teacher_cache, args.workers)
        train = WithTeacher(train, args.teacher_cache)
    train_loader = DataLoader(
        train, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
        sampler=RandomSampler(train, replacement=True, num_samples=args.steps_per_epoch * args.batch))
    val_loader = DataLoader(ProbeImages(args.data, val_rows, sphere, blank),
                            batch_size=8, num_workers=args.workers)
    objects = object_inputs(args.data)

    teacher_result = evaluate(teacher, teacher, val_loader)
    print(f"TouchNet fp32  {report(teacher_result)}", flush=True)

    model = UNet(args.base).cuda().to(memory_format=torch.channels_last)
    print(f"U-Net base{args.base}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.steps_per_epoch,
                                                pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for x, label, has_label, *target in train_loader:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            label, has_label = label.cuda(non_blocking=True), has_label.cuda(non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x).float()
            per_item = ((pred - label) ** 2).mean(dim=(1, 2, 3))
            loss = args.label_weight * (per_item * has_label).sum() / has_label.sum().clamp(min=1)
            if target:
                loss = loss + args.distill_weight * F.mse_loss(pred, target[0].cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        model.eval()
        result, object_result = evaluate(model, teacher, val_loader), evaluate_objects(model, teacher, objects)
        print(f"epoch {epoch:<2} loss={np.mean(losses):.4f} {report(result, object_result)} "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"base": args.base, "state_dict": model.state_dict()}, output)

    print(f"\nTouchNet fp32       {report(teacher_result)}")
    print(f"U-Net base{args.base} final {report(result, object_result)} -> {output}")


if __name__ == "__main__":
    main()
