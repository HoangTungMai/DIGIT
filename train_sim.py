"""Train TouchNet or the U-Net on FOTS-rendered frames of our DIGIT (make_fots_data.py).

Labels are exact for every shape: the gradient of the rendered gel height map, in py3DCal's
convention (channels = -d(depth)/dx, -d(depth)/dy, depth in px), so depth = -poisson(gx, gy)
as for the py3DCal models. Input is py3DCal's: (frame - background) / 255 in the frames' BGR
order plus 1-based x/y pixel coordinates. Training frames get a random reference drift
(smooth colour cast, column stripes, offset; see drift()) seen in our real frames.

Each epoch reports depth RMSE / peak error (px) on:
  sim-val   held-out frames of the training meshes (new poses),
  sim-test  frames of meshes never trained on (data/fots_sim/info.json "test_meshes"),
  real      FOTS's 40 real sphere frames of our DIGIT with fitted height maps
            (FOTS/outputs/digit_calibration/dataset.npz); "real-test" is its 8-frame test
            split, the frames FOTS's colour model was not fitted on.
Real is the number that matters: sim scores only show how well FOTS's own renderer is inverted.

  python train_sim.py --arch unet --epochs 15
  python train_sim.py --arch touchnet --init pretrained --batch 8 --lr 2e-4 --epochs 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

import py3DCal as p3d
from py3DCal.model_training.lib.add_coordinate_embeddings import add_coordinate_embeddings

from train_fast_touchnet import ROOT, depth
from unet import UNet

DEFAULT_REAL = ROOT.parent / "FOTS" / "outputs" / "digit_calibration" / "dataset.npz"


def gradients(height):
    """py3DCal labels of a depth map (px): (-d/dx, -d/dy)."""
    gy, gx = np.gradient(height.astype(np.float32))
    return np.stack([-gx, -gy])


def network_input(image, background):
    diff = (image.astype(np.float32) - background) / 255.0
    return add_coordinate_embeddings(torch.from_numpy(np.ascontiguousarray(diff.transpose(2, 0, 1))))


def drift(h, w, smooth=12.0, stripes=3.0, offset=4.0):
    """Random reference drift (BGR, /255) like our DIGIT's frames vs a stale background.

    Real frames differ from the background outside the contact by a smooth colour cast
    (~12-18/255 std), vertical column stripes and a global offset (exposure / white balance);
    FOTS renders none of that. Amplitudes are maxima, drawn uniformly per frame.
    """
    amp = torch.rand(3) * torch.tensor([smooth, stripes, offset]) / 255
    field = F.interpolate(torch.randn(1, 3, 4, 3) * amp[0], size=(h, w), mode="bicubic", align_corners=True)[0]
    columns = F.avg_pool1d(torch.randn(3, 1, w + 2) * amp[1], 3, 1)[:, 0]  # (3, w), slightly smoothed
    return field + columns[:, None, :] + torch.randn(3, 1, 1) * amp[2]


class SimFrames(Dataset):
    def __init__(self, root, split, augment=False):
        self.root, self.split, self.augment = root, split, augment
        self.background = np.load(root / "background.npy").astype(np.float32)
        self.n = np.load(root / f"{split}_depth.npy", mmap_mode="r").shape[0]
        self.images = self.depths = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.images is None:  # open in each worker
            self.images = np.load(self.root / f"{self.split}_images.npy", mmap_mode="r")
            self.depths = np.load(self.root / f"{self.split}_depth.npy", mmap_mode="r")
        height = np.asarray(self.depths[i], np.float32)
        x = network_input(self.images[i], self.background)
        if self.augment:
            x[:3] += drift(*height.shape)
        return x, torch.from_numpy(gradients(height)), torch.from_numpy(height)


class RealSpheres(Dataset):
    """FOTS's annotated real sphere frames; heights smoothed like the sim labels (sigma 2 px)."""

    def __init__(self, path, sigma=2.0, only=None):
        data = np.load(path)
        keep = np.ones(len(data["images"]), bool) if only is None else data["splits"] == only
        self.images, self.background = data["images"][keep], data["background"].astype(np.float32)
        self.heights = np.stack([cv2.GaussianBlur(h.astype(np.float32), (0, 0), sigma) for h in data["heights"][keep]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        h = self.heights[i]
        return network_input(self.images[i], self.background), torch.from_numpy(gradients(h)), torch.from_numpy(h)


def build(arch, init, base):
    if arch == "unet":
        return UNet(base)
    model = p3d.models.TouchNet()
    if init == "pretrained":  # py3DCal's weights for its own DIGIT: a starting point only
        model.load_weights(str(ROOT / "models" / "py3dcal" / "digit_pretrained_weights.pth"))
    return model


@torch.inference_mode()
def evaluate(model, loader):
    rmse, peak, grad = [], [], []
    for x, g, h in loader:
        pred = model(x.cuda()).float().cpu()
        grad.append(F.mse_loss(pred, g).item())
        for p, t in zip(pred.numpy(), h.numpy()):
            d = depth(p)
            rmse.append(np.sqrt(np.mean((d - t) ** 2)))
            peak.append(abs(d.max() - t.max()))
    return {"rmse": float(np.mean(rmse)), "peak": float(np.mean(peak)), "grad_mse": float(np.mean(grad))}


def report(results):
    return " | ".join(f"{k} rmse={v['rmse']:.2f} peak={v['peak']:.2f}" for k, v in results.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=("unet", "touchnet"), required=True)
    parser.add_argument("--init", choices=("scratch", "pretrained"), default="scratch",
                        help="TouchNet only: start from py3DCal's weights (trained on another DIGIT)")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fots_sim")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--base", type=int, default=32, help="U-Net width")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-augment", action="store_true", help="Train without the reference-drift augmentation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"{args.arch}_fots.pth"

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    train_all = SimFrames(args.data, "train")
    n_val = min(args.val_images, len(train_all) // 10)
    train = Subset(SimFrames(args.data, "train", augment=not args.no_augment), range(len(train_all) - n_val))
    sim_val = Subset(train_all, range(len(train_all) - n_val, len(train_all)))
    sim_test = SimFrames(args.data, "test")
    sim_test = Subset(sim_test, np.linspace(0, len(sim_test) - 1, min(args.val_images, len(sim_test))).astype(int))
    evals = {"sim-val": sim_val, "sim-test": sim_test,
             "real": RealSpheres(args.real), "real-test": RealSpheres(args.real, only="test")}
    info = json.loads((args.data / "info.json").read_text())
    print(f"{len(train)} train frames ({len(info['train_meshes'])} meshes); "
          + ", ".join(f"{k} {len(v)}" for k, v in evals.items())
          + f"; sim-test meshes: {', '.join(info['test_meshes'])}", flush=True)
    loaders = {k: DataLoader(v, batch_size=8, num_workers=min(args.workers, 4)) for k, v in evals.items()}
    train_loader = DataLoader(
        train, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
        sampler=RandomSampler(train, replacement=True, num_samples=args.steps_per_epoch * args.batch))

    model = build(args.arch, args.init, args.base).cuda().to(memory_format=torch.channels_last)
    print(f"{args.arch} ({args.init if args.arch == 'touchnet' else f'base{args.base}'}): "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    if args.arch == "touchnet" and args.init == "pretrained":
        model.eval()
        print(f"before training  {report({k: evaluate(model, l) for k, l in loaders.items()})}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.steps_per_epoch,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler()
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for x, g, _ in train_loader:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            g = g.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x).float()
            loss = F.mse_loss(pred, g)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        model.eval()
        results = {k: evaluate(model, l) for k, l in loaders.items()}
        print(f"epoch {epoch:<2} loss={np.mean(losses):.4f} {report(results)} ({time.monotonic() - started:.0f}s)",
              flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": args.arch, "base": args.base, "state_dict": model.state_dict(),
                    "data": str(args.data), "colour_order": info["colour_order"]}, output)
    print(f"\nfinal {args.arch} {report(results)} -> {output}")


if __name__ == "__main__":
    main()
