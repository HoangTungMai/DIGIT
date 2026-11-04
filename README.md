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
| `unet.py` | U-Net alternative to TouchNet: same 5-channel input and 2 gradient outputs, ~20× less compute |
| `make_fots_data.py` | Renders a labelled dataset of *our* DIGIT with [FOTS](https://github.com/Rancho-zhao/FOTS) (any shape, exact depth) |
| `train_sim.py` | Trains the U-Net or TouchNet on that dataset; scores unseen shapes and real sphere frames |
| `train_unet.py` | U-Net experiments on py3DCal's sphere data (labels, composites, distillation) |
| `models/unet_b32_fots.pth` | U-Net (base 32) trained on 20k FOTS frames of our DIGIT |
| `models/unet_b32_distill.pth` | U-Net (base 32) distilled from TouchNet on py3DCal's DIGIT data |
| `capture_real.py` | Records a reference plus labelled raw frames from a DIGIT for offline model comparison |
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

## U-Net trained on FOTS (`unet.py`, `make_fots_data.py`, `train_sim.py`)

TouchNet keeps all 9 convolutions at full resolution (~298 GMAC per frame). `unet.py` is a
4-level U-Net (base 32, 7.8M parameters, ~14 GMAC) with the same input and output, so the
same `fast_poisson` integration applies. It is ~9× faster in fp16 (1.7 ms vs 16 ms on the
RTX 2000 Ada).

**Why not py3DCal's data.** Every py3DCal label is a 2 mm probe sphere. Trained on those
labels, the U-Net, which sees almost the whole frame, learns to draw spheres. It scores
better than TouchNet on held-out probes but fails on other shapes: the pill in
`test_objects` comes out as two spheres. Adding summed multi-probe frames with summed labels
(`train_unet.py --composite-labels`) does not fix it. TouchNet avoids the shortcut because
its receptive field is only ~35 px. Distilling TouchNet does fix it
(`train_unet.py --distill-weight 1 --label-weight 0.5`), but then the U-Net only copies
TouchNet, on a DIGIT that is not ours.

`models/unet_b32_distill.pth` is that distilled model: 15 epochs, half the inputs summed
multi-probe frames with TouchNet as the only target. On py3DCal's data it is 0.27 / 1.22 px
RMSE / peak error from the held-out probe labels (TouchNet: 0.49 / 2.13). On the test
objects it is within 0.45–0.67 px of TouchNet, with the same shapes. On the real sphere
frames of our DIGIT it does worse than its teacher (2.90 / 9.4 px vs 2.71 / 6.8 px), so
under a sensor change the U-Net degrades more than TouchNet. The checkpoint holds
`{"base", "state_dict"}`, with the same input convention as below.

**FOTS dataset.** [FOTS](https://github.com/Rancho-zhao/FOTS) renders a DIGIT image from a
gel height map, using a normal-to-colour MLP and shadows calibrated on our sensor (4.7 mm
ball, 18.43 px/mm). `make_fots_data.py` presses the 21 STL indenters from
`FOTS/assets/daniel` into the gel, 1–2 per frame, each with a random x/y/z scale, yaw,
offset and 0.2–2.4 mm press. It adds the measured sensor noise and JPEG compression, and
saves the exact depth label. Torus, pacman, dots and wave1 are held out as unseen shapes.
`train_sim.py` also adds random reference drift (colour cast, column stripes), which real
frames show against a stale reference.

```bash
../FOTS/.venv/bin/python make_fots_data.py --train 20000 --test 2000   # ~4 min, 7.3 GB in data/fots_sim
python train_sim.py --arch unet --epochs 15 --output models/unet_b32_fots.pth   # ~30 min
python train_sim.py --arch touchnet --init pretrained --batch 8 --lr 2e-4 --epochs 8   # ~70 min
```

Depth RMSE / peak error in px. "Unseen shapes" is 500 FOTS frames of the 4 held-out meshes.
"Real spheres" is the 8 real frames of our DIGIT from FOTS's calibration set
(`FOTS/outputs/digit_calibration/dataset.npz`) that FOTS's colour model was not fitted on;
their peak depth is ~38 px:

| Model | Training frames | Unseen shapes (sim) | Real spheres |
|---|---|---|---|
| **U-Net, FOTS (`models/unet_b32_fots.pth`)** | 240k | **2.28 / 2.6** | **2.43 / 5.6** |
| U-Net, FOTS, at epoch 4 | 64k | 2.51 / 3.3 | 2.61 / 9.6 |
| TouchNet fine-tuned on FOTS from py3DCal's weights | 64k | 3.68 / 6.7 | 4.68 / 23.0 |
| TouchNet, py3DCal weights (another DIGIT) | — | 5.64 / 8.7 | 2.71 / 6.8 |

TouchNet costs ~58 ms per training frame vs ~7 ms for the U-Net, so it got fewer frames.
At the same 64k frames the U-Net is still clearly ahead, and it is ahead at every
gradient-steepness level. Both reproduce the unseen shapes. TouchNet's errors are mostly in
depth magnitude: on the real spheres it predicts about half the depth.

Limitations: the unseen shapes are simulated, and the only real frames with ground truth are
spheres. FOTS models no gel mechanics, and its shadow fit is rough. Pressing a few printed
indenters of known shape on the sensor is the next real test.

Checkpoints hold `{"arch", "base", "state_dict", "colour_order"}`. Load them with
`UNet(c["base"]).load_state_dict(c["state_dict"])`. The input is `(frame - reference) / 255`
in OpenCV BGR order plus 1-based x/y pixel coordinates, as for TouchNet. `digit.py` does not
load the U-Net yet.
