# DIGIT realtime depth

Live depth reconstruction for a [DIGIT](https://digit.ml/) tactile sensor using
[py3DCal](https://rohankotanu.github.io/3DCal/)'s pretrained TouchNet. Target platform:
Jetson AGX Orin (JetPack 5.1.2, Python 3.8); also runs on a CUDA workstation or on CPU.

TouchNet predicts surface gradients from the difference between the current frame and an
unloaded reference frame; `py3DCal.fast_poisson` integrates them into a depth map.
Depth is in **model pixel units**, not millimetres.

## Files

| File | Purpose |
|---|---|
| `digit.py` | Live viewer: camera, reference collection, realtime depth, freeze/render, `.npz` export |
| `touchnet_fp16.py` | TouchNet rewritten for fp16 CUDA inference (used by `digit.py` on CUDA) |
| `touchnet_trt.py` | TouchNet as a TensorRT engine, fp16 or INT8 (library only, see below) |
| `fast_touchnet.py` | FastTouchNet: TouchNet with conv4–conv6 at reduced resolution |
| `train_fast_touchnet.py` | Fine-tunes FastTouchNet from TouchNet (distillation + probe-sphere labels) |
| `models/fast_touchnet_s2.pth` | FastTouchNet /2 checkpoint, fine-tuned on ~1.1k probe images |
| `requirements-digit.txt` | Jetson install steps (header) and pip requirements |

## Installation

**Jetson AGX Orin:** follow the numbered steps in the header of
[`requirements-digit.txt`](requirements-digit.txt). In short: NVIDIA's CUDA torch wheel
(PyPI's aarch64 torch is CPU-only), torchvision built from source, the requirements file,
then `py3DCal==1.1.0` with `--no-deps`. Keep py3DCal pinned: `digit.py` imports one of its
internal modules.

**Workstation:** install a CUDA build of torch and torchvision, then

```bash
pip install -r requirements-digit.txt
pip install --no-deps py3DCal==1.1.0
```

The pretrained DIGIT weights are downloaded to `models/py3dcal/` on first run.

## Usage

```bash
python digit.py --list          # list connected sensors
python digit.py                 # live view with realtime depth
```

At start-up the camera is stabilised for 90 frames, then 30 frames are averaged into the
reference. **Keep the gel unloaded until "Reference ready" is printed.**

| Key | Action |
|---|---|
| `R` | Freeze: render depth for the frame at keypress |
| `L` | Resume live depth |
| `B` | Re-collect the reference (unload the gel first) |
| `Q` / `Esc` | Quit |

Useful options (`python digit.py --help` for all):

| Option | Effect |
|---|---|
| `--serial SN` | Choose a sensor when several are connected |
| `--device {auto,cpu,cuda,mps}` | Inference device (default: CUDA if available) |
| `--fp32` | Disable fp16 on CUDA (slower, reference accuracy) |
| `--manual` | Start frozen; render only on `R` |
| `--render-once --no-show --output out.npz` | Headless: render one frame, save frame/reference/depth |
| `--depth-range 20` | Colour-map range of the depth panel, in model pixels |
| `--weights file.pth` | Custom TouchNet weights |
| `--model-color-order rgb` | For custom weights trained on RGB images (the published DIGIT weights are BGR) |

On a Jetson, set maximum performance before measuring speed:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

## fp16 inference (`touchnet_fp16.py`)

On CUDA, `digit.py` runs `TouchNetFP16` instead of py3DCal's module under autocast. It
loads the same weights and changes only how they are executed:

- **BatchNorm folded into the convolutions**, in fp64, then cast to fp16 once.
- **Coordinate channels precomputed.** conv1's x/y inputs (up to 320) produce large values
  that BatchNorm then nearly cancels, which costs most of fp16's precision. Their
  contribution plus the bias is a fixed map, computed once in fp64, so conv1 only sees the
  3 image channels.
- **fp16 weights in channels_last layout** (tensor-core kernels).
- **Preprocessing on the GPU** from the two uint8 frames, via pinned memory.
- **One CUDA graph** replays the whole network (the frame size is fixed).

Result: about 2.5× faster than fp32, with depth within ~0.001 px of fp32. The old
autocast path was ~0.01 px off. These figures are approximate; re-measure on your
hardware, especially on Jetson, where the CPU-side `fast_poisson` may dominate frame time.

## FastTouchNet

`fast_touchnet.py` runs conv4–conv6 (~84% of the compute) at 1/`scale` resolution, which
makes the network roughly 2× faster at `scale=2`. Fine-tune it with the py3DCal DIGIT
calibration dataset extracted to `data/digit_calibration_data/`
(`probe_images/`, `blank_images/`, `annotations/`):

```bash
python train_fast_touchnet.py --scale 2 --epochs 10 --steps-per-epoch 1000
```

The best epoch (lowest depth RMSE against TouchNet) is saved to `models/fast_touchnet_s<scale>.pth`.

`TouchNetFP16(state_dict, scale=2)` accepts these checkpoints, but `digit.py` does not
load them yet: `--weights` expects a plain TouchNet file.

## TensorRT (`touchnet_trt.py`)

Builds fp16 or INT8 engines (INT8 for conv2–conv8, with calibration from real frame pairs)
and runs them with `TouchNetTRT`. Engines are specific to the GPU and TensorRT version, so
build them on the device that runs them. It is not wired into `digit.py`, and the build
script it refers to (`build_touchnet_trt.py`) is not in the repository yet.
