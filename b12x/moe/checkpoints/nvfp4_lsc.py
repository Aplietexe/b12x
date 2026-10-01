"""TP-independent NVFP4-LSC storage with exact E4M3 byte slicing."""

from __future__ import annotations

from contextlib import ExitStack
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from b12x._lib.quant.nvfp4_lsc import make_nvfp4_lsc_batch
from b12x.moe.fused_moe.weights import Nvfp4LscWeights, PackedWeights

SCHEMA = "lil-nvfp4-lsc-checkpoint/1"
CODEC = "byte-window4-fixed-stream-u24-exceptions/1"


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the compressed container and retained source tensor inventory."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    contract = json.loads((root / "build-contract.json").read_text())
    for key, expected in (
        ("schema", SCHEMA),
        ("codec", CODEC),
        ("family", "glm53_nvfp4"),
    ):
        if manifest.get(key) != expected or contract.get(key) != expected:
            raise ValueError(f"NVFP4-LSC requires {key}={expected!r}")
    names = contract["source_names"]
    files = {item["file"] for item in manifest["shards"]}
    if set(names.values()) != files:
        raise ValueError("NVFP4-LSC manifest and source tensor inventory disagree")
    for name in files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("NVFP4-LSC shard names must be checkpoint-local")
    return contract


def slice_scale_plane(fixed, exceptions, rows, columns, row_slice, column_slice):
    """Copy aligned compressed extents and rebase exception positions."""
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("NVFP4-LSC row slices require complete 16-row slabs")
    if not (0 <= c0 < c1 <= columns and c0 % 2 == c1 % 2 == columns % 2 == 0):
        raise ValueError("NVFP4-LSC column slices require complete nibble pairs")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("NVFP4-LSC requires uint8 fixed and uint32 exception tensors")
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + columns // 2))
    bases = stream[:, :16]
    if np.any(bases > 240):
        raise ValueError("NVFP4-LSC row bases must be in 0..240")
    packed = stream[:, 16:].reshape(rows, columns // 2)
    selected = packed[r0:r1, c0 // 2 : c1 // 2]
    result = np.concatenate(
        (bases[r0 // 16 : r1 // 16], selected.reshape((r1 - r0) // 16, -1)), 1
    )
    words = exceptions.numpy().reshape(-1)
    positions = words & np.uint32(0xFFFFFF)
    if len(words) and (
        positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("NVFP4-LSC exception positions must be sorted and unique")
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    local_positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
    selected_words = (words[keep] & np.uint32(0xFF000000)) | local_positions
    return result.copy(), selected_words.astype(np.uint32)


def read_nvfp4_lsc_layer(
    root,
    layer_index,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
):
    """Load one GLM expert layer, preserving native weights and calibration."""
    root = Path(root)
    contract = checkpoint_contract(str(root.resolve()))
    if (num_experts, hidden_size, intermediate_size) != (288, 4096, 2048):
        raise ValueError("GLM NVFP4-LSC requires 288 experts and H4096/I2048")
    if tp_size not in (1, 2, 4, 8, 16) or not 0 <= tp_rank < tp_size:
        raise ValueError("GLM NVFP4-LSC supports aligned TP1/2/4/8/16 extents")
    if not 3 <= layer_index < 45:
        raise ValueError("GLM NVFP4-LSC encodes routed layers 3 through 44")
    names = contract["source_names"]
    local = intermediate_size // tp_size
    first, last = tp_rank * local, (tp_rank + 1) * local
    w13 = torch.empty((num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8)
    w2 = torch.empty((num_experts, hidden_size, local // 2), dtype=torch.uint8)
    # Model loaders may set BF16 as the default dtype. Calibration belongs to
    # the source FP32 contract and must not be rounded with the model weights.
    g13, g2 = (torch.empty(num_experts, dtype=torch.float32) for _ in range(2))
    a13, a2 = (torch.empty(num_experts, dtype=torch.float32) for _ in range(2))
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []
    with ExitStack() as stack:
        handles = {}

        def handle(name):
            filename = names[name]
            if filename not in handles:
                handles[filename] = stack.enter_context(
                    safe_open(root / "tensors" / filename, framework="pt", device="cpu")
                )
            return handles[filename]

        def scalar(name):
            value = handle(name).get_tensor(name)
            if value.numel() != 1 or value.dtype != torch.float32:
                raise ValueError(f"NVFP4 calibration must be one FP32 value: {name}")
            if not bool(torch.isfinite(value).all() and (value > 0).all()):
                raise ValueError(
                    f"NVFP4 calibration must be positive and finite: {name}"
                )
            return value.reshape(())

        for expert in range(num_experts):
            f13, e13, global13, input13 = [], [], [], []
            for matrix, projection in enumerate(("up_proj", "gate_proj", "down_proj")):
                stem = (
                    f"model.language_model.layers.{layer_index}.mlp."
                    f"experts.{expert}.{projection}"
                )
                name, scale = stem + ".weight", stem + ".weight_scale"
                view = handle(name).get_slice(name)
                expected = (
                    [intermediate_size, hidden_size // 2]
                    if matrix < 2
                    else [hidden_size, intermediate_size // 2]
                )
                if view.get_shape() != expected or view.get_dtype() != "U8":
                    raise ValueError(f"NVFP4 nibble geometry/dtype mismatch: {name}")
                global_scale, input_scale = (
                    scalar(scale + "_2"),
                    scalar(stem + ".input_scale"),
                )
                if matrix < 2:
                    w13[expert, matrix * local : (matrix + 1) * local].copy_(
                        view[first:last, :]
                    )
                    rows, columns = intermediate_size, hidden_size // 16
                    row_slice, column_slice = (first, last), (0, columns)
                    global13.append(global_scale)
                    input13.append(input_scale)
                else:
                    w2[expert].copy_(view[:, first // 2 : last // 2])
                    rows, columns = hidden_size, intermediate_size // 16
                    row_slice, column_slice = (0, rows), (first // 16, last // 16)
                    g2[expert], a2[expert] = global_scale, input_scale
                reader = handle(scale)
                fixed, exceptions = slice_scale_plane(
                    reader.get_tensor(scale + ".nvfp4_lsc_fixed"),
                    reader.get_tensor(scale + ".nvfp4_lsc_exceptions"),
                    rows,
                    columns,
                    row_slice,
                    column_slice,
                )
                if matrix < 2:
                    f13.append(fixed)
                    if matrix:
                        exceptions += np.uint32(local * columns)
                    e13.append(exceptions)
                else:
                    fixed2.append(fixed)
                    exceptions2.append(exceptions)
            if not torch.equal(global13[0], global13[1]):
                raise ValueError(
                    "NVFP4-LSC requires equal gate/up global weight scales"
                )
            g13[expert] = global13[0]
            a13[expert] = torch.stack(input13).amax()
            fixed13.append(np.concatenate(f13))
            exceptions13.append(np.concatenate(e13))
    batch13 = make_nvfp4_lsc_batch(
        fixed13, exceptions13, rows=2 * local, columns=hidden_size // 16, device=device
    )
    batch2 = make_nvfp4_lsc_batch(
        fixed2, exceptions2, rows=hidden_size, columns=local // 16, device=device
    )
    return Nvfp4LscWeights(
        packed=PackedWeights(
            w13=w13.to(device),
            w2=w2.to(device),
            w13_block_scales=w13_scale_scratch,
            w2_block_scales=w2_scale_scratch,
            w13_global_scales=g13.to(device),
            w2_global_scales=g2.to(device),
            input_scale=a13.to(device).reciprocal(),
            intermediate_scale=a2.to(device).reciprocal(),
            immutable_input_scales=True,
        ),
        w13_scales=batch13,
        w2_scales=batch2,
    )
