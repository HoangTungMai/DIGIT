"""Render a labelled DIGIT dataset with FOTS (https://github.com/Rancho-zhao/FOTS).

FOTS turns a gel height map into a DIGIT image with an MLP (normal -> colour) and calibrated
shadows. The FOTS calibration in --fots/--calibration was fitted on *our* DIGIT, so this
renders that sensor, unlike py3DCal's dataset (another DIGIT). Every frame comes with its
exact gel height map, so any shape gets ground truth, not only spheres.

Shapes are the 21 STL indenters in FOTS/assets/daniel, each 16x16 mm. Every frame draws 1-2
of them with a random anisotropic x/y scale, relief (z) scale, yaw and offset, presses them
0.2-2.4 mm into the gel (with FOTS's gel-pad curvature), and renders it, with shadows in
--shadow-fraction of the frames. Sensor
noise (sigma ~1-2.5/255 measured on our DIGIT) and, half the time, JPEG compression are added
to the image; the depth label stays clean.

--test-meshes are never used for the train split: the test split shows shapes the networks
have not seen (sim generalisation); the real check is FOTS's own 40 real sphere frames
(see train_sim.py).

Output (--output dir): {split}_images.npy uint8 (N,320,240,3) in FOTS/OpenCV BGR order,
{split}_depth.npy float16 (N,320,240) gel depth in px (positive into the gel, smoothed as
FOTS smooths it before rendering), {split}_meta.csv, background.npy (the reference frame:
network input is image - background) and info.json.

Run with the FOTS environment (open3d):
  ../FOTS/.venv/bin/python make_fots_data.py --train 20000 --test 2000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_FOTS = ROOT.parent / "FOTS"
DEFAULT_TEST_MESHES = ("torus", "pacman", "dots", "wave1")
GRID_MM = 0.04  # resolution of each mesh's precomputed top surface


def mesh_surface(path, o3d):
    """Top surface z(x, y) in mm of a mesh on a GRID_MM grid centred on its x/y bounds.

    Pixels the mesh does not cover are NaN. Ray casting once per mesh lets every frame resample
    the surface with an affine warp instead of casting rays again.
    """
    mesh = o3d.io.read_triangle_mesh(str(path))
    lo, hi = mesh.get_min_bound(), mesh.get_max_bound()
    mesh.translate([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0])
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2
    n = int(math.ceil(2 * half / GRID_MM)) + 1
    coords = (np.arange(n) - (n - 1) / 2) * GRID_MM
    ray_z = hi[2] + 1.0
    rays = np.zeros((n, n, 6), np.float32)
    rays[..., 0], rays[..., 1] = coords[None, :], coords[:, None]
    rays[..., 2], rays[..., 5] = ray_z, -1
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    surface = (ray_z - t).astype(np.float32)
    surface[~np.isfinite(t)] = np.nan
    return surface - np.nanmax(surface)  # top at 0


def place(surface, mm_to_px, h, w, scale_x, scale_y, yaw_deg, x_mm, y_mm):
    """Resample a mesh surface onto the h x w sensor grid (sensor centre = mesh centre + offset)."""
    n = surface.shape[0]
    # sensor px -> mesh grid index: centre, translate, rotate, unscale, to grid units
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    k = 1 / (mm_to_px * GRID_MM)
    a = np.array([[c / scale_x, s / scale_x], [-s / scale_y, c / scale_y]]) * k
    centre_px = np.array([w / 2 + x_mm * mm_to_px, h / 2 + y_mm * mm_to_px])
    offset = (n - 1) / 2 - a @ centre_px
    m = np.hstack([a, offset[:, None]]).astype(np.float32)
    valid = np.isfinite(surface)
    z = cv2.warpAffine(np.where(valid, surface, 0), m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cover = cv2.warpAffine(valid.astype(np.float32), m, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(cover > 0.5, z, -np.inf).astype(np.float32)


def sample_frame(rng, surfaces, names, pad_mm, mm_to_px, h, w):
    """(depth_mm, params): 1-2 randomly posed meshes pressed into the gel."""
    depth = np.zeros((h, w), np.float32)
    params = []
    for _ in range(1 if rng.random() < 0.75 else 2):
        i = int(rng.integers(len(names)))
        p = dict(mesh=names[i], scale_x=float(rng.uniform(0.4, 1.3)), scale_y=0.0,
                 z_scale=float(rng.uniform(0.5, 1.5)), yaw=float(rng.uniform(0, 360)),
                 x_mm=float(rng.uniform(-4, 4)), y_mm=float(rng.uniform(-5, 5)),
                 press_mm=float(rng.uniform(0.2, 2.4)))
        p["scale_y"] = float(p["scale_x"] * np.exp(rng.uniform(-0.4, 0.4)))
        top = place(surfaces[i], mm_to_px, h, w, p["scale_x"], p["scale_y"], p["yaw"], p["x_mm"], p["y_mm"])
        top = top * p["z_scale"] + pad_mm
        if not np.isfinite(top).any():
            continue
        depth = np.maximum(depth, np.maximum(top - top.max() + p["press_mm"], 0))
        params.append(p)
    return depth, params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fots", type=Path, default=DEFAULT_FOTS, help="FOTS repository")
    parser.add_argument("--calibration", type=Path, help="FOTS calibration bundle (default FOTS/outputs/digit_calibration)")
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--test-meshes", default=",".join(DEFAULT_TEST_MESHES))
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--shadow-fraction", type=float, default=0.7,
                        help="Share of frames rendered with FOTS shadows (its shadow fit is rough, ~26 px rms)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fots_sim")
    parser.add_argument("--preview", type=int, default=0, help="Only write a preview grid of N frames")
    args = parser.parse_args()
    fots = args.fots.resolve()
    calibration = (args.calibration or fots / "outputs" / "digit_calibration").resolve()

    sys.path.insert(0, str(fots))
    import open3d as o3d
    import torch
    import params as pr
    from tactile_render import get_simapproach

    torch.set_num_threads(4)
    sim = get_simapproach("digit", device="cuda" if torch.cuda.is_available() else "cpu",
                          calibration_dir=calibration)
    mm_to_px = pr.mm_to_pixel = sim.mm_to_pixel
    h, w = pr.sensor_h, pr.sensor_w
    pad_mm = np.load(fots / "utils" / "utils_data" / "digit_pad.npy").astype(np.float32)
    background = np.asarray(sim.background, np.float32)

    test_names = [n for n in args.test_meshes.split(",") if n]
    paths = sorted((fots / "assets" / "daniel").glob("*.stl"))
    missing = set(test_names) - {p.stem for p in paths}
    if missing:
        parser.error(f"Unknown test meshes: {sorted(missing)}")
    print(f"Ray casting {len(paths)} meshes...", flush=True)
    surfaces = {p.stem: mesh_surface(p, o3d) for p in paths}
    splits = {"train": [n for n in surfaces if n not in test_names], "test": test_names}

    rng = np.random.default_rng(args.seed)
    jobs = [("preview", args.preview)] if args.preview else [("train", args.train), ("test", args.test)]
    args.output.mkdir(parents=True, exist_ok=True)
    for split, count in jobs:
        names = splits["train" if split == "preview" else split]
        if split == "preview":
            names = list(surfaces)
        surf = [surfaces[n] for n in names]
        if split != "preview":
            images = np.lib.format.open_memmap(args.output / f"{split}_images.npy", "w+", np.uint8, (count, h, w, 3))
            depths = np.lib.format.open_memmap(args.output / f"{split}_depth.npy", "w+", np.float16, (count, h, w))
        tiles, meta, started = [], [], time.monotonic()
        for i in range(count):
            params = []
            while not params:
                depth_mm, params = sample_frame(rng, surf, names, pad_mm, mm_to_px, h, w)
            height = depth_mm * mm_to_px
            mask = height > 0
            shadow = not args.no_shadow and rng.random() < args.shadow_fraction
            image = sim.generate(height, mask, shadow=shadow).astype(np.float32)
            noise_sigma = float(rng.uniform(0.5, 2.5))
            image += rng.normal(0, noise_sigma, image.shape).astype(np.float32)
            image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
            jpeg = int(rng.integers(70, 96)) if rng.random() < 0.5 else 0
            if jpeg:
                image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
            label = sim.smooth_heightMap(height, mask)  # the surface FOTS rendered
            meta.append({"index": i, "shadow": shadow, "noise_sigma": noise_sigma, "jpeg": jpeg,
                         "meshes": "+".join(p["mesh"] for p in params), "params": json.dumps(params)})
            if split == "preview":
                d = cv2.applyColorMap(np.clip(label / max(label.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
                                      cv2.COLORMAP_VIRIDIS)
                t = np.hstack([image, d])
                cv2.putText(t, meta[-1]["meshes"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                tiles.append(cv2.resize(t, None, fx=0.5, fy=0.5))
                continue
            images[i], depths[i] = image, label
            if (i + 1) % 1000 == 0 or i + 1 == count:
                rate = (i + 1) / (time.monotonic() - started)
                print(f"{split}: {i + 1}/{count} ({rate:.0f} frames/s)", flush=True)
        if split == "preview":
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
            grid = np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)])
            cv2.imwrite(str(args.output / "preview.png"), grid)
            print(f"wrote {args.output / 'preview.png'}")
            return
        images.flush(), depths.flush()
        del images, depths
        pd.DataFrame(meta).to_csv(args.output / f"{split}_meta.csv", index=False)

    np.save(args.output / "background.npy", background)
    (args.output / "info.json").write_text(json.dumps({
        "fots": str(fots), "calibration": str(calibration), "mm_to_pixel": mm_to_px,
        "colour_order": "bgr", "height": h, "width": w, "shadow_fraction": 0.0 if args.no_shadow else args.shadow_fraction, "seed": args.seed,
        "train_meshes": splits["train"], "test_meshes": splits["test"],
        "counts": {"train": args.train, "test": args.test},
        "depth": "px, positive into the gel; FOTS-smoothed height map the image was rendered from",
    }, indent=2) + "\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
