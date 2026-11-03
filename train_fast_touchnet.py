"""Fine-tune FastTouchNet (TouchNet with conv4-6 at 1/scale resolution) from TouchNet weights.

Loss = MSE to the original TouchNet's gradients (distillation) + MSE to the probe-sphere
labels from py3DCal's DIGIT calibration data. Images stream from disk, so the full
dataset (~36k images) fits; the last 10% of probe positions is held out for validation.

  myenv/bin/python train_fast_touchnet.py --scale 2 --epochs 10 --steps-per-epoch 1000
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

import py3DCal as p3d
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings
from py3DCal.model_training.lib.precompute_gradients import precompute_gradients

from fast_touchnet import FastTouchNet

ROOT = Path(__file__).resolve().parent


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def shift_sphere(sphere, x, y):
    """py3DCal's get_gradient_map without re-reading the CSV for every image."""
    h, w, _ = sphere.shape
    dx, dy = int(x) - w // 2, int(y) - h // 2
    pad = max(abs(dx), abs(dy))
    out = np.pad(sphere, ((pad, pad), (pad, pad), (0, 0)))
    out = np.roll(np.roll(out, dx, axis=1), dy, axis=0)
    return out[pad:pad + h, pad:pad + w]


class ProbeImages(Dataset):
    def __init__(self, data, rows, sphere, blank):
        self.data, self.rows, self.sphere, self.blank = data, rows, sphere, blank

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        image = load(self.data / "probe_images" / row.img_name)
        x = torch.from_numpy(np.ascontiguousarray((image - self.blank).transpose(2, 0, 1)))
        label = shift_sphere(self.sphere, row.x_px, row.y_px).transpose(2, 0, 1)
        return add_coordinate_embeddings(x), torch.from_numpy(label.astype(np.float32))


def split_rows(data, val_fraction, val_limit):
    blank = load(data / "blank_images" / "blank.png")
    h, w, _ = blank.shape
    ann = pd.read_csv(data / "annotations" / "annotations.csv")
    have = {p.name for p in (data / "probe_images").glob("*.png")}
    rows = ann[ann.img_name.isin(have) & ann.x_px.between(0, w) & ann.y_px.between(0, h)]
    rows = rows.assign(n=rows.img_name.str.split("_").str[0].astype(int)).sort_values("n")
    split = int(len(rows) * (1 - val_fraction))
    val = rows.iloc[split:]
    val = val.iloc[np.linspace(0, len(val) - 1, min(val_limit, len(val))).astype(int)]
    meta = pd.read_json(data / "annotations" / "metadata.json", typ="series")
    # precompute_gradients only opens the CSV's first image for its size.
    first = ROOT / "data" / ".first_available.csv"
    rows.head(1).to_csv(first, index=False)
    sphere = precompute_gradients(str(data), str(first), r=meta.probe_radius_mm * meta.px_per_mm)
    return rows.iloc[:split], val, sphere.astype(np.float32), blank


def depth(g):
    return np.clip(-p3d.fast_poisson(g[0], g[1]), 0, None)


@torch.inference_mode()
def evaluate(model, teacher, loader):
    grad_mse, distill_mse, depth_rmse, peak_err = [], [], [], []
    for x, label in loader:
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            pred, ref = model(x).float(), teacher(x).float()
        grad_mse.append(F.mse_loss(pred, label.cuda()).item())
        distill_mse.append(F.mse_loss(pred, ref).item())
        for p, t in zip(pred.cpu().numpy(), ref.cpu().numpy()):
            dp, dt = depth(p), depth(t)
            depth_rmse.append(np.sqrt(np.mean((dp - dt) ** 2)))
            peak_err.append(abs(dp.max() - dt.max()))
    return tuple(float(np.mean(v)) for v in (grad_mse, distill_mse, depth_rmse, peak_err))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "digit_calibration_data")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="Random batches per epoch (the full set is ~4k batches of 8)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-weight", type=float, default=0.5,
                        help="Weight of the probe-sphere label loss; distillation weight is 1")
    parser.add_argument("--val-images", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--init", type=Path, help="Resume from a saved FastTouchNet checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"fast_touchnet_s{args.scale}.pth"

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    teacher = p3d.models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT,
                                  root=ROOT / "models" / "py3dcal").cuda().eval()
    train_rows, val_rows, sphere, blank = split_rows(args.data, 0.1, args.val_images)
    print(f"{len(train_rows)} train / {len(val_rows)} val images", flush=True)
    train = ProbeImages(args.data, train_rows, sphere, blank)
    train_loader = DataLoader(
        train, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True,
        sampler=RandomSampler(train, replacement=True, num_samples=args.steps_per_epoch * args.batch))
    val_loader = DataLoader(ProbeImages(args.data, val_rows, sphere, blank),
                            batch_size=args.batch, num_workers=args.workers)

    student = FastTouchNet(copy.deepcopy(teacher), args.scale).cuda()
    if args.init:
        student.load_state_dict(torch.load(args.init)["state_dict"])
    # Keep BatchNorm on its running stats: the probe data is narrow.
    student.eval()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    report = lambda r: (f"val grad_mse={r[0]:.4f} distill={r[1]:.4f} "
                        f"depth_rmse={r[2]:.3f}px peak_err={r[3]:.3f}px")
    best = evaluate(student, teacher, val_loader)
    print(f"epoch 0  {report(best)}", flush=True)
    best_state = copy.deepcopy(student.state_dict())
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        for x, label in train_loader:
            x, label = x.cuda(non_blocking=True), label.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                target = teacher(x).float()
            with torch.autocast("cuda", dtype=torch.float16):
                pred = student(x).float()
            loss = F.mse_loss(pred, target) + args.label_weight * F.mse_loss(pred, label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        result = evaluate(student, teacher, val_loader)
        mark = ""
        if result[2] < best[2]:
            best, best_state, mark = result, copy.deepcopy(student.state_dict()), " *"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"scale": args.scale, "state_dict": best_state}, output)
        print(f"epoch {epoch:<2} {report(result)} ({time.monotonic() - started:.0f}s){mark}", flush=True)

    teacher_val = evaluate(FastTouchNet(teacher, 1), teacher, val_loader)
    print(f"\nTouchNet (goc) val grad_mse={teacher_val[0]:.4f}")
    print(f"Best FastTouchNet /{args.scale} {report(best)} -> {output}")


if __name__ == "__main__":
    main()
