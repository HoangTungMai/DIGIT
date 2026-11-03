"""TouchNet as a TensorRT engine: fp16, or INT8 for conv2-conv8 (explicit Q/DQ).

Uses the same folded layers as touchnet_fp16.py (BatchNorm folded, conv1's coordinate
channels as a fixed fp64 map). In int8 mode conv2-conv8 (~99% of the compute) take INT8
inputs and weights: per-output-channel weight scales from the weights themselves, one
activation scale per layer from calibration images. conv1 (3 input channels) and conv9
(2 output channels) stay fp16.

Only TensorRT APIs that exist since 8.5 (JetPack 5.1.2) are used. Engines are specific to
the GPU and TensorRT version: build them on the device that runs them (build_touchnet_trt.py).
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from touchnet_fp16 import LAYERS, fold_layers

INT8_LAYERS = range(1, LAYERS - 1)  # 0-based: conv2..conv8


def prepare(frames):
    """(N, H, W, 3) uint8 [frame, reference] on the GPU -> (1, 3, H, W) float BGR difference / 255."""
    frame, reference = frames.permute(0, 3, 1, 2).float().unbind(0)
    return ((frame - reference) * (1.0 / 255.0))[None]


@torch.inference_mode()
def calibrate(state_dict, pairs, color_order="bgr", percentile=99.99, samples=20000, device="cuda"):
    """Activation |max| entering conv2..conv8, as a percentile of fp32 activations on real pairs.

    pairs: iterable of (frame, reference) HxWx3 uint8 BGR numpy arrays.
    Returns a list of 7 floats (one per INT8 layer).
    """
    pairs = list(pairs)
    h, w = pairs[0][0].shape[:2]
    layers, coord_bias = fold_layers(state_dict, h, w, color_order)
    layers = [(wt.float().to(device), None if b is None else b.float().to(device)) for wt, b in layers]
    coord_bias = coord_bias.float().to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    values = [[] for _ in INT8_LAYERS]
    for frame, reference in pairs:
        frames = torch.from_numpy(np.stack([frame, reference])).to(device)
        x = prepare(frames)
        for i, (wt, b) in enumerate(layers[:-1]):
            x = F.conv2d(x, wt, b, padding=wt.shape[-1] // 2)
            if i == 0:
                x = x + coord_bias
            x = F.relu(x)
            if i + 1 in INT8_LAYERS:
                flat = x.flatten()
                pick = torch.randint(flat.numel(), (samples,), device=device, generator=generator)
                values[i].append(flat[pick].cpu())
    return [float(np.percentile(torch.cat(v).numpy(), percentile)) for v in values]


def build_engine(state_dict, height, width, precision="int8", act_amax=None, color_order="bgr",
                 workspace_mb=1024):
    """Return a serialized TensorRT engine (bytes). act_amax (from calibrate) is needed for int8."""
    import tensorrt as trt

    if precision not in ("fp16", "int8"):
        raise ValueError("precision must be fp16 or int8")
    if precision == "int8" and (act_amax is None or len(act_amax) != len(INT8_LAYERS)):
        raise ValueError(f"int8 needs {len(INT8_LAYERS)} activation amax values (calibrate())")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    keep = []  # numpy arrays must outlive the build

    def const(array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        keep.append(array)
        return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def qdq(tensor, amax, axis=None):
        scale = const(np.maximum(np.asarray(amax, dtype=np.float32), 1e-8) / 127.0)
        q = network.add_quantize(tensor, scale)
        if axis is not None:  # set before add_dequantize, which validates q's scale against its axis
            q.axis = axis
        dq = network.add_dequantize(q.get_output(0), scale)
        if axis is not None:
            dq.axis = axis
        return dq.get_output(0)

    layers, coord_bias = fold_layers(state_dict, height, width, color_order)
    x = network.add_input("diff", trt.float32, (1, 3, height, width))
    for i, (w, b) in enumerate(layers):
        w = w.float().numpy()
        out_channels, k = w.shape[0], w.shape[-1]
        bias = trt.Weights()
        if b is not None:
            keep.append(np.ascontiguousarray(b.float().numpy()))
            bias = trt.Weights(keep[-1])
        if precision == "int8" and i in INT8_LAYERS:
            x = qdq(x, act_amax[i - 1])
            conv = network.add_convolution_nd(x, out_channels, (k, k), trt.Weights(), bias)
            conv.set_input(1, qdq(const(w), np.abs(w).reshape(out_channels, -1).max(1), axis=0))
        else:
            keep.append(w)
            conv = network.add_convolution_nd(x, out_channels, (k, k), trt.Weights(w), bias)
        conv.padding_nd = (k // 2, k // 2)
        x = conv.get_output(0)
        if i == 0:
            x = network.add_elementwise(x, const(coord_bias.numpy()), trt.ElementWiseOperation.SUM).get_output(0)
        if i < LAYERS - 1:
            x = network.add_activation(x, trt.ActivationType.RELU).get_output(0)
    x.name = "gradients"
    network.mark_output(x)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(engine)


def save_calibration(path, act_amax, percentile):
    with open(path, "w") as f:
        json.dump({"layers": "conv2..conv8 inputs", "percentile": percentile, "act_amax": act_amax}, f, indent=1)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)["act_amax"]


class TouchNetTRT:
    """Run a TouchNet engine: predict(frame, reference) -> (2, H, W) float32 gradients."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Cannot load TensorRT engine {engine_path} (built for another GPU/TensorRT?)")
        self.context = self.engine.create_execution_context()
        _, _, height, width = self.engine.get_tensor_shape("diff")
        self.size = (height, width)
        self.stream = torch.cuda.Stream(device)
        self.host_in = torch.empty((2, height, width, 3), dtype=torch.uint8).pin_memory()
        self.host_out = torch.empty((2, height, width), dtype=torch.float32).pin_memory()
        self.dev_in = torch.empty((2, height, width, 3), dtype=torch.uint8, device=device)
        self.diff = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
        self.out = torch.empty((1, 2, height, width), dtype=torch.float32, device=device)
        self.context.set_tensor_address("diff", self.diff.data_ptr())
        self.context.set_tensor_address("gradients", self.out.data_ptr())

    @torch.inference_mode()
    def predict(self, frame, reference):
        if frame.shape != (*self.size, 3) or reference.shape != frame.shape \
                or frame.dtype != np.uint8 or reference.dtype != np.uint8:
            raise ValueError(f"Frame and reference must be {self.size[0]}x{self.size[1]}x3 uint8")
        self.host_in[0].numpy()[:] = frame
        self.host_in[1].numpy()[:] = reference
        with torch.cuda.stream(self.stream):
            self.dev_in.copy_(self.host_in, non_blocking=True)
            self.diff.copy_(prepare(self.dev_in))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
            self.host_out.copy_(self.out[0], non_blocking=True)
        self.stream.synchronize()
        return self.host_out.numpy().copy()
