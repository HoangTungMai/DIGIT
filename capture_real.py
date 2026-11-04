"""Record raw DIGIT frames for offline model comparison (no model is run).

Uses the same camera path as digit.py (py3DCal's DIGIT class, digit-interface QVGA). Keep
the gel unloaded until "Reference ready": the reference is the mean of --reference-frames
frames, and the next --blank-frames unloaded frames are kept too (they show the sensor's
noise and drift). Then press an object on the gel and

  S       save the current frame (the last typed label is stored with it)
  0-9     set the label to that digit (e.g. 1 = sphere, 2 = screw; note them down)
  Q/Esc   write the .npz and quit

Output .npz: reference (H,W,3) uint8, blanks (K,H,W,3), frames (N,H,W,3), labels (N,),
times (N,) seconds since the reference, serial. All images are BGR, as digit.py uses them.

  python capture_real.py --output real_session1.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", help="Auto-select if only one DIGIT is connected")
    parser.add_argument("--output", type=Path, required=True, help="New .npz file")
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--reference-frames", type=int, default=30)
    parser.add_argument("--blank-frames", type=int, default=10)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz" or args.output.exists():
        parser.error("--output must be a new .npz file")

    from digit_interface import DigitHandler
    import py3DCal as p3d

    devices = [d["serial"] for d in DigitHandler.list_digits()]
    if not devices:
        parser.error("No DIGIT sensor found")
    if args.serial and args.serial not in devices:
        parser.error(f"DIGIT {args.serial} is not connected")
    if not args.serial and len(devices) > 1:
        parser.error("Multiple DIGIT sensors found; select one with --serial")
    serial = args.serial or devices[0]
    sensor = p3d.DIGIT(serial)
    sensor.connect()
    frames, labels, times, label = [], [], [], 0
    try:
        print(f"DIGIT {serial}: stabilising {args.warmup_frames} frames, keep gel UNLOADED", flush=True)
        sensor.flush_frames(args.warmup_frames)
        stack = np.stack([sensor.capture_image() for _ in range(args.reference_frames)]).astype(np.float64)
        reference = np.rint(stack.mean(0)).astype(np.uint8)
        blanks = np.stack([sensor.capture_image() for _ in range(args.blank_frames)])
        start = time.monotonic()
        print("Reference ready. Press an object, S to save, 0-9 to set the label, Q to quit.", flush=True)
        while True:
            frame = sensor.capture_image()
            diff = np.clip((frame.astype(np.int16) - reference) * 3 + 128, 0, 255).astype(np.uint8)
            view = np.hstack([frame, diff])
            cv2.putText(view, f"label {label} | saved {len(frames)}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("DIGIT capture | S save | 0-9 label | Q quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if ord("0") <= key <= ord("9"):
                label = key - ord("0")
                print(f"label = {label}", flush=True)
            if key in (ord("s"), ord("S")):
                frames.append(frame.copy())
                labels.append(label)
                times.append(time.monotonic() - start)
                print(f"saved #{len(frames)} (label {label})", flush=True)
    finally:
        sensor.disconnect()
        cv2.destroyAllWindows()
    if not frames:
        print("Nothing saved.")
        return
    with args.output.open("xb") as f:
        np.savez_compressed(f, reference=reference, blanks=blanks, frames=np.stack(frames),
                            labels=np.array(labels), times=np.array(times), serial=serial)
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
