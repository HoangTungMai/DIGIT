"""TouchNet re-defined for fp16 CUDA inference.

Same weights as py3DCal's TouchNet (or a FastTouchNet checkpoint), but:
  * BatchNorm folded into the convolutions (folded in fp64, then cast to fp16),
  * conv1's coordinate-embedding channels precomputed as a fixed fp64 bias map
    (in plain fp16 they cause ~5x the depth error of the rest of the network),
  * weights stored in fp16 and activations in channels_last (tensor-core layout),
  * image preparation (BGR difference / 255 + coordinate embeddings) on the GPU from uint8,
  * the whole pipeline replayed as one CUDA graph (frame size is fixed).

Dropout is dropped: it is the identity at inference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = 9


def fold_batchnorm(state_dict):
    """Return [(weight, bias)] for conv1..conv9 with bn1..bn8 folded in, as fp64 tensors."""
    sd = {k[len("net."):] if k.startswith("net.") else k: v.detach().cpu().double() for k, v in state_dict.items()
          if not k.endswith("num_batches_tracked")}
    layers = []
    for i in range(1, LAYERS + 1):
        w, b = sd[f"conv{i}.weight"], sd[f"conv{i}.bias"]
        if i < LAYERS:
            scale = sd[f"bn{i}.weight"] / torch.sqrt(sd[f"bn{i}.running_var"] + 1e-5)
            w = w * scale[:, None, None, None]
            b = (b - sd[f"bn{i}.running_mean"]) * scale + sd[f"bn{i}.bias"]
        layers.append((w, b))
    return layers


def fold_layers(state_dict, height, width, color_order="bgr"):
    """fold_batchnorm, plus conv1 split into its 3 image channels and a fixed coordinate map.

    conv1 sees [B, G, R, x, y]. The x/y coordinate channels (up to 320) give large pre-BN
    values that the folded bias then mostly cancels, which loses most of fp16's precision.
    Convolution is linear, so their contribution plus the bias is a fixed (1, 32, H, W) map:
    it is computed once in fp64, and conv1 keeps only the image channels (and no bias).
    Returns (layers, coord_bias), all fp64 on the CPU.
    """
    if color_order not in ("bgr", "rgb"):
        raise ValueError("color_order must be bgr or rgb")
    layers = fold_batchnorm(state_dict)
    w1, b1 = layers[0]
    xs = torch.arange(1, width + 1, dtype=torch.float64).expand(height, width)
    ys = torch.arange(1, height + 1, dtype=torch.float64)[:, None].expand(height, width)
    coord_bias = F.conv2d(torch.stack([xs, ys])[None], w1[:, 3:], b1, padding=w1.shape[-1] // 2)
    w1 = w1[:, :3]
    if color_order == "rgb":  # weights trained on RGB: feed our BGR difference to flipped weights
        w1 = w1.flip(1)
    layers[0] = (w1, None)
    return layers, coord_bias


class TouchNetFP16(torch.nn.Module):
    """Gradients (2, H, W) from a DIGIT frame and its unloaded reference, both HxWx3 uint8 BGR.

    scale > 1 runs conv4-conv6 at 1/scale resolution (FastTouchNet checkpoints).
    color_order="rgb" flips channels for weights trained on RGB images.
    """

    def __init__(self, state_dict, height=320, width=240, scale=1, color_order="bgr",
                 device="cuda", cuda_graph=True):
        super().__init__()
        self.scale = scale
        self.size = (height, width)
        layers, coord_bias = fold_layers(state_dict, height, width, color_order)
        cl = torch.channels_last
        self.coord_bias = coord_bias.to(device, torch.float16).contiguous(memory_format=cl)
        self.weights = [w.to(device, torch.float16).contiguous(memory_format=cl) for w, _ in layers]
        self.biases = [None if b is None else b.to(device, torch.float16) for _, b in layers]
        # Static buffers: pinned host <-> device copies, reused by the CUDA graph.
        self.host_in = torch.empty((2, height, width, 3), dtype=torch.uint8).pin_memory()
        self.host_out = torch.empty((2, height, width), dtype=torch.float32).pin_memory()
        self.dev_in = torch.empty((2, height, width, 3), dtype=torch.uint8, device=device)
        self.graph = None
        self.dev_out = None
        if cuda_graph:
            self._capture()

    def network(self, x):
        """fp16 (1, 3, H, W) channels_last BGR difference -> fp16 gradients."""
        w, b = self.weights, self.biases
        x = F.relu(F.conv2d(x, w[0], padding=w[0].shape[-1] // 2) + self.coord_bias, inplace=True)
        for i in range(1, 3):
            x = F.relu(F.conv2d(x, w[i], b[i], padding=w[i].shape[-1] // 2), inplace=True)
        if self.scale > 1:
            x = F.avg_pool2d(x, self.scale)
        for i in range(3, 6):
            x = F.relu(F.conv2d(x, w[i], b[i], padding=w[i].shape[-1] // 2), inplace=True)
        if self.scale > 1:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        for i in range(6, 8):
            x = F.relu(F.conv2d(x, w[i], b[i], padding=w[i].shape[-1] // 2), inplace=True)
        return F.conv2d(x, w[8], b[8])

    def forward(self, frames):
        """frames: (2, H, W, 3) uint8 on device, [frame, reference] -> (2, H, W) float32."""
        # HWC -> CHW permute of a contiguous HWC tensor is already channels_last.
        frame, reference = frames.permute(0, 3, 1, 2).half().unbind(0)
        diff = (frame - reference) * (1.0 / 255.0)  # integer difference is exact in fp16
        return self.network(diff[None])[0].float()

    @torch.inference_mode()
    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # cudnn algorithm selection before capture
                self.forward(self.dev_in)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.dev_out = self.forward(self.dev_in)

    @torch.inference_mode()
    def predict(self, frame, reference):
        """numpy HxWx3 uint8 BGR frame + reference -> numpy (2, H, W) float32 gradients."""
        if frame.shape != (*self.size, 3) or reference.shape != frame.shape \
                or frame.dtype != np.uint8 or reference.dtype != np.uint8:
            raise ValueError(f"Frame and reference must be {self.size[0]}x{self.size[1]}x3 uint8")
        self.host_in[0].numpy()[:] = frame
        self.host_in[1].numpy()[:] = reference
        self.dev_in.copy_(self.host_in, non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
            out = self.dev_out
        else:
            out = self.forward(self.dev_in)
        self.host_out.copy_(out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.host_out.numpy().copy()
