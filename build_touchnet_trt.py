"""Build a TensorRT engine for DIGIT TouchNet (run it on the device that will use the engine).

INT8 needs activation ranges. Either calibrate from py3DCal's DIGIT data (256 images spread
over the first 90% of probe positions, the training part of train_fast_touchnet.py), which also
writes --calibration, or reuse a saved --calibration JSON (no dataset needed, e.g. on the Jetson):

  python build_touchnet_trt.py --data data/digit_calibration_data      # writes calibration + engine
  python build_touchnet_trt.py                                          # reuses models/touchnet_int8_calib.json
  python build_touchnet_trt.py --precision fp16

Then: python digit.py --engine models/touchnet_int8.engine
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import touchnet_trt as tt

ROOT = Path(__file__).resolve().parent
# digit.py downloads the weights to models/py3dcal; the Zenodo download lives in data/zenodo.
WEIGHTS = [ROOT / "models" / "py3dcal" / "digit_pretrained_weights.pth",
           ROOT / "data" / "zenodo" / "digit_pretrained_weights.pth"]


def load(path):
    # Keep the stored channel values: the DIGIT PNGs hold BGR data (see digit.prepare_model_image).
    return np.asarray(Image.open(path).convert("RGB"))


def calibration_pairs(data, count):
    blank = load(data / "blank_images" / "blank.png")
    files = sorted((data / "probe_images").glob("*.png"), key=lambda p: int(p.name.split("_")[0]))
    if not files:
        raise SystemExit(f"No probe images in {data / 'probe_images'}")
    train = files[:int(len(files) * 0.9)]
    picks = np.linspace(0, len(train) - 1, min(count, len(train))).astype(int)
    return [(load(train[i]), blank) for i in picks]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path,
                        default=next((p for p in WEIGHTS if p.is_file()), WEIGHTS[0]),
                        help="TouchNet .pth state dict (py3DCal DIGIT weights by default)")
    parser.add_argument("--precision", choices=("int8", "fp16"), default="int8")
    parser.add_argument("--data", type=Path, help="py3DCal DIGIT calibration data: recalibrate INT8 ranges")
    parser.add_argument("--calibration", type=Path, default=ROOT / "models" / "touchnet_int8_calib.json")
    parser.add_argument("--calib-images", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--int8-convs", default="".join(map(str, tt.DEFAULT_INT8_CONVS)),
                        help="Convolutions run in INT8, as digits (default: %(default)s)")
    parser.add_argument("--height", type=int, default=320,
                        help="Frame rows (DIGIT frames are portrait 320x240; taken from --data if given)")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--color-order", choices=("bgr", "rgb"), default="bgr")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"touchnet_{args.precision}.engine"
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")

    state_dict = torch.load(args.weights, map_location="cpu")
    amax = None
    if args.precision == "int8":
        if args.data:
            pairs = calibration_pairs(args.data, args.calib_images)
            print(f"Calibrating on {len(pairs)} images (p{args.percentile})...", flush=True)
            args.height, args.width = pairs[0][0].shape[:2]
            amax = tt.calibrate(state_dict, pairs, args.color_order, args.percentile)
            args.calibration.parent.mkdir(parents=True, exist_ok=True)
            tt.save_calibration(args.calibration, amax, args.percentile)
            print(f"Saved calibration to {args.calibration}")
        elif args.calibration.is_file():
            amax = tt.load_calibration(args.calibration)
        else:
            parser.error(f"INT8 needs --data or an existing --calibration ({args.calibration})")
    convs = tuple(int(c) for c in args.int8_convs)
    print(f"Building {args.precision} engine for {args.height}x{args.width} frames"
          + (f" (INT8 conv{args.int8_convs})" if amax else "") + "; this takes a minute...", flush=True)
    engine = tt.build_engine(state_dict, args.height, args.width, args.precision, amax,
                             args.color_order, int8_convs=convs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(engine)
    print(f"Wrote {output} ({len(engine) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
