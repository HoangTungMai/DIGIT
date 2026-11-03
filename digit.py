"""Live DIGIT view with realtime depth; R freezes the frame at keypress, L resumes live.

Keep the gel unloaded during reference collection. B resets the reference;
Q/Esc stops. Depth uses model pixel units, not millimetres.
Model documentation: https://rohankotanu.github.io/3DCal/
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from pathlib import Path
import time

import cv2
import numpy as np


def prepare_model_image(frame, reference, color_order="bgr"):
    """Match py3DCal DIGIT's recorded channel convention, not display RGB.

    Calibrator.probe() writes Image.fromarray(sensor.capture_image()) directly.
    DIGIT.capture_image() returns OpenCV BGR, so the published PNGs retain BGR
    values even though PIL labels those channels RGB. Keep BGR for those weights.
    """
    if (frame.shape != reference.shape or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or reference.dtype != np.uint8):
        raise ValueError("Frame and reference must be matching HxWx3 uint8 BGR images")
    if color_order not in ("bgr", "rgb"):
        raise ValueError("Model color order must be bgr or rgb")
    image = frame.astype(np.float32) / 255.0
    blank = reference.astype(np.float32) / 255.0
    difference = image - blank
    if color_order == "rgb":
        difference = difference[:, :, ::-1]
    return np.ascontiguousarray(difference.transpose(2, 0, 1))


class DigitDepthModel:
    """TouchNet inference using the DIGIT calibration recording convention."""

    def __init__(self, root, device="cpu", weights=None, color_order="bgr", half=False):
        import torch
        import py3DCal as p3d
        from py3DCal.model_training.lib.add_coordinate_embeddings import (
            add_coordinate_embeddings,
        )

        self.torch = torch
        self.p3d = p3d
        self.add_coordinate_embeddings = add_coordinate_embeddings
        self.device = device
        self.color_order = color_order
        # fp16 (TouchNetFP16: folded BN, CUDA graph) is ~2.5x faster than fp32 and its depth
        # differs from true fp32 by ~0.001 px; see touchnet_fp16.py.
        self.half = half and device == "cuda"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True  # frame size never changes
        if weights is None:
            root.mkdir(parents=True, exist_ok=True)
            self.model = p3d.models.TouchNet(
                load_pretrained=True, sensor_type=p3d.SensorType.DIGIT, root=root,
            )
        else:
            self.model = p3d.models.TouchNet()
            self.model.load_weights(weights)
        self.model.to(device).eval()
        self.fast = None  # TouchNetFP16, built on the first frame (needs the frame size)

    def predict(self, frame, reference):
        if self.half:
            if self.fast is None:
                from touchnet_fp16 import TouchNetFP16
                self.fast = TouchNetFP16(self.model.state_dict(), *frame.shape[:2],
                                         color_order=self.color_order, device=self.device)
            gradients = self.fast.predict(frame, reference)
            depth = self.p3d.fast_poisson(gradients[0], gradients[1])
            return np.clip(-depth, 0, None).astype(np.float32)
        image = self.torch.from_numpy(prepare_model_image(frame, reference, self.color_order))
        image = self.add_coordinate_embeddings(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode(), self.torch.autocast(
                "cuda", dtype=self.torch.float16, enabled=self.half):
            gradients = self.model(image)[0].float().cpu().numpy()
        depth = self.p3d.fast_poisson(gradients[0], gradients[1])
        return np.clip(-depth, 0, None).astype(np.float32)


def predict_frame(model, frame, reference):
    started = time.monotonic()
    depth = model.predict(frame, reference)
    return frame, reference, depth, time.monotonic() - started


class DepthRenderer:
    """One inference worker (max_workers=1), keeping only the latest request."""

    def __init__(self, model, executor):
        self.model = model
        self.executor = executor
        self.request_id = 0
        self.completed = 0
        self.future = None
        self.snapshot = None
        self.result = None
        self.done_times = deque(maxlen=10)

    def request(self, frame, reference, live=False):
        if live:
            # Keep showing the previous depth until this frame finishes.
            if self.future is not None:
                self.future.cancel()
        else:
            self.reset()
        self.request_id += 1
        self.snapshot = frame.copy()
        self.future = self.executor.submit(predict_frame, self.model,
                                           self.snapshot, reference.copy())
        return self.request_id

    def reset(self):
        # cancel() drops a waiting job; a running one finishes in the background
        # (the single worker prevents overlap) but its result is discarded.
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.snapshot = None
        self.result = None
        self.done_times.clear()

    def update(self):
        if self.future is None or not self.future.done():
            return False
        self.result = self.future.result()
        self.future = None
        self.completed += 1
        self.done_times.append(time.monotonic())
        return True

    def fps(self):
        if len(self.done_times) < 2:
            return 0.0
        return (len(self.done_times) - 1) / (self.done_times[-1] - self.done_times[0])

    def live_status(self):
        if self.result is None:
            return "Live: starting... | R: freeze | B: reference | Q: quit"
        _, _, depth, elapsed = self.result
        return (f"Live: {self.fps():.1f} fps, {elapsed * 1000:.0f} ms/depth | "
                f"max={depth.max():.2f}px | R: freeze | B: reference | Q: quit")

    def status(self):
        if self.future is not None:
            if self.future.running():
                return f"Rendering R #{self.request_id}... | R: replace with next frame"
            return f"R #{self.request_id} waiting | R: replace waiting frame"
        if self.result is not None:
            _, _, depth, elapsed = self.result
            return f"R #{self.request_id}: {elapsed:.1f}s | max={depth.max():.2f}px | R: again | L: live"
        return "Frozen | R: render depth | L: live | B: reference | Q: quit"


def draw_preview(frame, snapshot, depth, status, depth_range, snapshot_title):
    snapshot = np.zeros_like(frame) if snapshot is None else snapshot
    if depth is None:
        colored = np.zeros_like(frame)
    else:
        colored = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / depth_range),
                                    cv2.COLORMAP_VIRIDIS)
    panels = []
    for image, title in zip(
        (frame, snapshot, colored),
        ("DIGIT live", snapshot_title, f"Depth: 0..{depth_range:g} px"),
    ):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(panel, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    preview = np.hstack(panels)
    cv2.rectangle(preview, (0, preview.shape[0] - 25),
                  (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
    cv2.putText(preview, status, (5, preview.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List sensors and exit")
    parser.add_argument("--serial", help="Auto-select if only one DIGIT is connected")
    parser.add_argument("--model-root", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "py3dcal")
    parser.add_argument("--weights", type=Path, help="Optional custom TouchNet .pth")
    parser.add_argument("--model-color-order", choices=("bgr", "rgb"), default="bgr",
                        help="Published DIGIT weights use bgr; use rgb for custom RGB-trained weights")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable fp16 inference on CUDA (about 2x slower)")
    parser.add_argument("--manual", action="store_true",
                        help="Start frozen: render only on R instead of realtime")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--reference-frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=90,
                        help="Discard startup frames before reference collection (default: 3s at 30fps)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N displayed depth results; 0 runs until stopped")
    parser.add_argument("--render-once", action="store_true",
                        help="Capture one frame after reference collection, render and exit")
    parser.add_argument("--depth-range", type=float, default=20,
                        help="Fixed display range in model pixel units")
    parser.add_argument("--output", type=Path,
                        help="Save last displayed frame, reference and depth to a new .npz")
    parser.add_argument("--no-show", action="store_true",
                        help="Headless mode; requires --render-once")
    args = parser.parse_args()
    if (args.reference_frames < 1 or args.warmup_frames < 0 or args.frames < 0
            or args.threads < 1 or not np.isfinite(args.depth_range) or args.depth_range <= 0):
        parser.error("Invalid reference, warmup, frame count, threads or depth range")
    if args.no_show and not args.render_once and not args.list:
        parser.error("--no-show requires --render-once; R needs a preview window")
    if args.output and (args.output.suffix.lower() != ".npz" or args.output.exists()):
        parser.error("--output must be a new .npz file")
    if args.weights and not args.weights.is_file():
        parser.error(f"Weights file does not exist: {args.weights}")
    try:
        from digit_interface import DigitHandler
    except ImportError as exc:
        parser.exit(1, f"Missing sensor dependency: {exc}\n"
                    "Install: see the header of requirements-digit.txt\n")
    devices = {device["serial"]: device for device in DigitHandler.list_digits()}
    if args.list:
        for serial in devices:
            print(f"Connected DIGIT Serial Number: {serial}")
        if not devices:
            print("No DIGIT sensor connected.")
        return
    if not devices:
        parser.error("No DIGIT sensor found. Check the USB connection.")
    if args.serial and args.serial not in devices:
        parser.error(f"DIGIT {args.serial} is not connected")
    if not args.serial and len(devices) > 1:
        parser.error("Multiple DIGIT sensors found; select one with --serial")
    serial = args.serial or next(iter(devices))
    try:
        import torch
        import py3DCal as p3d
    except ImportError as exc:
        parser.exit(1, f"Missing model dependency: {exc}\n"
                    "Install: see the header of requirements-digit.txt\n")
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading py3DCal TouchNet for DIGIT {serial} on {device}...", flush=True)
    model = DigitDepthModel(args.model_root, device, args.weights, args.model_color_order,
                            half=not args.fp32)
    sensor = p3d.DIGIT(serial)
    executor = ThreadPoolExecutor(max_workers=1)
    renderer = DepthRenderer(model, executor)
    window = "DIGIT | R: freeze | L: live | B: reference | Q/Esc: stop"
    reference_sum = None
    reference_count = 0
    reference = None
    live = not (args.manual or args.render_once)
    prediction_limit = 1 if args.render_once else args.frames
    with ExitStack() as stack:
        # Python 3.8 has no shutdown(cancel_futures=); reset() (run first, LIFO)
        # cancels the only job that can be waiting.
        stack.callback(executor.shutdown, wait=True)
        stack.callback(renderer.reset)
        try:
            sensor.connect()
            stack.callback(sensor.disconnect)
            if not args.no_show:
                stack.callback(cv2.destroyAllWindows)
            print(f"Stabilizing camera for {args.warmup_frames} frames; keep gel UNLOADED.", flush=True)
            sensor.flush_frames(args.warmup_frames)
            if device != "cpu":
                # First CUDA/cuDNN call is slow; pay it now, not on the first depth frame.
                frame = sensor.capture_image()
                model.predict(frame, frame)
            print(f"Keep gel UNLOADED for {args.reference_frames} reference frames. "
                  + ("Realtime depth starts after that; R freezes a frame." if live
                     else "Then press R to render depth for the frame at keypress."), flush=True)
            while True:
                frame = sensor.capture_image()
                # Poll even during recalibration so stale jobs can finish safely.
                if renderer.update() and not live:
                    _, _, depth, elapsed = renderer.result
                    print(f"R #{renderer.request_id}: {elapsed:.2f} s/depth | "
                          f"max={depth.max():.2f} px", flush=True)
                ready = reference_count >= args.reference_frames
                if not ready:
                    if reference_sum is None:
                        reference_sum = np.zeros_like(frame, dtype=np.float64)
                    reference_sum += frame
                    reference_count += 1
                    reference = np.rint(reference_sum / reference_count).astype(np.uint8)
                    status = f"Keep unloaded: reference {reference_count}/{args.reference_frames}"
                    if reference_count == args.reference_frames:
                        reference_sum = None
                        print("Reference ready." + (" Realtime depth running." if live else
                              " Press R to render; model idle until requested."), flush=True)
                elif live:
                    # Feed the newest frame as soon as the worker is free.
                    if renderer.future is None:
                        renderer.request(frame, reference, live=True)
                    status = renderer.live_status()
                else:
                    if args.render_once and renderer.request_id == 0:
                        renderer.request(frame, reference)
                    status = renderer.status()
                if not args.no_show:
                    if renderer.result is None:
                        depth, snapshot = None, renderer.snapshot
                    else:
                        snapshot, _, depth, _ = renderer.result
                    if live:
                        title = "Depth input frame"
                    elif renderer.snapshot is not None:
                        snapshot, title = renderer.snapshot, f"R #{renderer.request_id} sensor frame"
                    else:
                        title = "Press R to capture"
                    cv2.imshow(window, draw_preview(frame, snapshot, depth, status,
                                                    args.depth_range, title))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("l"), ord("L")) and not live:
                        live = True
                        renderer.reset()
                        print("Realtime depth resumed.", flush=True)
                    if key in (ord("r"), ord("R")):
                        if ready:
                            live = False
                            request_id = renderer.request(frame, reference)
                            print(f"R #{request_id}: frame at keypress captured.", flush=True)
                        else:
                            print("Reference is still calibrating. Wait before pressing R.", flush=True)
                    if key in (ord("b"), ord("B")):
                        renderer.reset()
                        reference_sum = None
                        reference_count = 0
                        reference = None
                if prediction_limit and renderer.completed >= prediction_limit:
                    break
        except KeyboardInterrupt:
            print("\nStopping DIGIT.", flush=True)
    if args.output and renderer.result is not None:
        frame, reference, depth, _ = renderer.result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            np.savez_compressed(output, frame=frame, reference=reference, depth=depth,
                                serial=serial, depth_units="model_pixels",
                                request_id=renderer.request_id)
        print(f"Saved last displayed prediction to {args.output}", flush=True)
    print(f"Finished: {renderer.completed} depth renders.", flush=True)


if __name__ == "__main__":
    main()
