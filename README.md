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
| `touchnet_trt.py` | TouchNet as a TensorRT engine, fp16 or INT8 (`digit.py --engine`) |
| `build_touchnet_trt.py` | Calibrates INT8 and builds the TensorRT engine on the target device |
| `models/touchnet_int8_calib.json` | INT8 activation ranges from the DIGIT calibration data (lets the Jetson build without the dataset) |
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
| `--engine models/touchnet_int8.engine` | Run a TensorRT engine instead (INT8: ~2× faster than fp16, see below) |
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

Result: about 2.5× faster than fp32 and ~1.3× faster than the old autocast path, with
depth within ~0.001 px of fp32 (the autocast path was ~4× further off); see the table under
INT8 TensorRT. `fast_poisson` on the CPU adds ~1–2 ms per frame on a workstation; re-measure
on the Jetson, whose CPU is slower.

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

## INT8 TensorRT (`touchnet_trt.py`, `build_touchnet_trt.py`)

Same folded layers as `TouchNetFP16`, built as a TensorRT network with explicit INT8
quantization (Q/DQ) of conv3–conv8 (~97% of the compute): per-output-channel weight scales,
one activation scale per layer at the 99.99th percentile of fp32 activations on 256 training
images. conv1, conv2 and conv9 stay fp16. conv2 is kept in fp16 on purpose: its input is a
small contact signal on top of large coordinate-driven activations, and quantizing it alone
triples the depth error while saving almost no time.

Engines are specific to the GPU and TensorRT version, so build on the device that runs them:

```bash
python build_touchnet_trt.py --data data/digit_calibration_data   # recalibrate + build
python build_touchnet_trt.py                                       # reuse models/touchnet_int8_calib.json
python digit.py --engine models/touchnet_int8.engine
```

On the Jetson, TensorRT comes with JetPack (step 6 in `requirements-digit.txt`). Only
TensorRT 8.5 APIs are used, but the code has so far only been run with TensorRT 10.13.

Measured on an RTX 2000 Ada (sm_89), 320×240 frames, 300 held-out probe images (last 10% of
probe positions), depth compared with true fp32 (TF32 off):

| Path | Network ms | vs `--fp32` (TF32) | vs true fp32 | Depth RMSE | Peak error |
|---|---|---|---|---|---|
| fp32, TF32 off | 91 | 0.4× | 1× | — | — |
| `digit.py --fp32` (PyTorch default: TF32) | ~40 | 1× | 2.3× | | |
| py3DCal TouchNet, autocast fp16 (old) | 22.1 | 1.8× | 4.1× | 0.0017 px | 0.0062 px |
| `TouchNetFP16` | 16.1 | 2.5× | 5.7× | 0.0003 px | 0.0008 px |
| TensorRT fp16 | 14.3 | 2.8× | 6.4× | 0.0017 px | 0.0056 px |
| **TensorRT INT8 conv3–8 (default)** | **7.2** | **5.5×** | **12.6×** | 0.026 px | 0.072 px |
| TensorRT INT8 conv4–8 (`--int8-convs 45678`) | 8.1 | 4.9× | 11.2× | 0.014 px | 0.040 px |
| TensorRT INT8 conv2–8 | 7.2 | 5.5× | 12.6× | 0.075 px | 0.214 px |

PyTorch runs fp32 convolutions as TF32 on Ampere and newer GPUs (the Orin included) unless
`torch.backends.cudnn.allow_tf32 = False`, so `digit.py --fp32` is TF32, not true fp32.

For scale: fp32 TouchNet itself is 0.34 px RMSE / 1.22 px peak error from the probe-sphere
labels, so INT8 adds little to the model's own error, and no quantization-aware training was
needed. End to end in `digit.py` (including `fast_poisson`): 41 ms/frame `--fp32`, 18.3 ms
fp16, 8.3 ms INT8 (~4.9× faster than `--fp32`). Ratios on the Orin will differ: re-measure there.

Lower precisions are not an option on the Orin: its Ampere GPU has no FP8 tensor cores
(Ada/Hopper and later only), FP4 is Blackwell-only, and int16 convolution is not supported
by cuDNN at all.
