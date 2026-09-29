"""Read independently transformed matrices from trellis-dense-checkpoint/1.

Each matrix stores native K16/N16 tile words and independent input/output
scale vectors. Tensor parallelism partitions complete H128 intermediate
blocks. The adapter only permutes codeword bytes and slices FP16 vectors;
it never decodes, rounds, or requantizes weights.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu
from b12x.moe.fused_moe.config import TrellisConfig
from b12x.moe.fused_moe.source import TrellisExtent, TrellisSource
from b12x.moe.fused_moe.weights import ScaleFactors, TrellisWeights


@lru_cache(maxsize=4)
def _manifest(root: str) -> dict:
    directory = Path(root)
    manifest = json.loads((directory / "trellis-manifest.json").read_text())
    required = {
        "schema": "trellis-dense-checkpoint/1",
        "codebook": "lut_e4m3",
        "bits_per_quantized_coefficient": 2,
        "state_bits": 16,
        "tailbite_context": 128,
        "tile_shape": [16, 16],
        "hadamard_block": 128,
        "scale_dtype": "float16",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"independent trellis requires {key}={value!r}")
    tables_name = manifest["decoder_tables"]
    if Path(tables_name).name != tables_name:
        raise ValueError("decoder_tables must name a checkpoint-local file")
    tables_path = directory / tables_name
    digest = hashlib.sha256(tables_path.read_bytes()).hexdigest()
    if digest != manifest["decoder_tables_sha256"]:
        raise ValueError("independent trellis decoder table hash mismatch")
    with safe_open(tables_path, framework="pt", device="cpu") as handle:
        lut = handle.get_tensor("lut_e4m3_k2")
        if not torch.equal(lut, lut_e4m3_direct_table_cpu()[:65536]):
            raise ValueError("checkpoint K2 lookup differs from the B12X decoder")
        h = torch.ones((1, 1), dtype=torch.float64, device="cpu")
        while h.shape[0] < 128:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        h = (h / 128**0.5).float()
        if not torch.equal(handle.get_tensor("hadamard_128"), h):
            raise ValueError("checkpoint does not use the normalized Sylvester H128")
    by_name = {item["name"]: item for item in manifest["quantized"]}
    if len(by_name) != len(manifest["quantized"]):
        raise ValueError("independent trellis manifest has duplicate matrix names")
    manifest["by_name"] = by_name
    return manifest


def read_independent_layer(
    root: str | Path,
    layer_index: int,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> tuple[TrellisSource, TrellisWeights]:
    """Read one TP shard, retaining per-expert gate/up/down transforms."""
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid tensor-parallel rank or size")
    if (
        hidden_size <= 0
        or intermediate_size <= 0
        or hidden_size % 128
        or intermediate_size % (128 * tp_size)
    ):
        raise ValueError("TP shards must contain complete 128-channel transforms")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    root = Path(root)
    manifest = _manifest(str(root.resolve()))
    local = intermediate_size // tp_size
    first = tp_rank * local
    tiles = hidden_size // 16
    slots = local // 32
    # One row stores two N16 (FC1) or K16 (FC2) tiles per matrix/expert.
    codes = torch.empty(
        (slots, num_experts, 3, 2, tiles, 32), dtype=torch.int16, device="cpu"
    )
    inputs = torch.empty(
        (num_experts, 2, hidden_size), dtype=torch.float16, device="cpu"
    )
    middle = torch.empty((num_experts, 3, local), dtype=torch.float16, device="cpu")
    outputs = torch.empty((num_experts, hidden_size), dtype=torch.float16, device="cpu")
    filename = f"trellis-layer-{layer_index:02d}.safetensors"
    with safe_open(root / filename, framework="pt", device="cpu") as handle:
        for expert in range(num_experts):
            for matrix, projection in enumerate(("w1", "w3", "w2")):
                stem = f"layers.{layer_index}.ffn.experts.{expert}.{projection}"
                shape = (
                    [intermediate_size, hidden_size]
                    if matrix < 2
                    else [hidden_size, intermediate_size]
                )
                entry = manifest["by_name"].get(stem + ".weight")
                if entry is None or entry["logical_shape"] != shape:
                    raise ValueError(f"manifest geometry mismatch for {stem}")
                if entry["packed_file"] != filename:
                    raise ValueError(f"manifest shard mismatch for {stem}")
                view = handle.get_slice(stem + ".trellis")
                if view.get_shape() != [shape[1] // 16, shape[0] // 16, 32]:
                    raise ValueError(f"native tile shape mismatch for {stem}")
                if view.get_dtype() != "I16":
                    raise TypeError(f"{stem}.trellis must contain int16 tile words")
                suh = handle.get_tensor(stem + ".suh")
                svh = handle.get_tensor(stem + ".svh")
                if (
                    suh.dtype != torch.float16
                    or svh.dtype != torch.float16
                    or suh.shape != (shape[1],)
                    or svh.shape != (shape[0],)
                ):
                    raise ValueError(f"FP16 transform vector mismatch for {stem}")
                if not (torch.isfinite(suh).all() and torch.isfinite(svh).all()):
                    raise ValueError(f"nonfinite transform vector for {stem}")
                if matrix < 2:
                    packed = view[:, first // 16 : (first + local) // 16, :]
                    codes[:, expert, matrix].copy_(
                        packed.reshape(tiles, slots, 2, 32).permute(1, 2, 0, 3)
                    )
                    inputs[expert, matrix].copy_(suh)
                    middle[expert, matrix].copy_(svh[first : first + local])
                else:
                    packed = view[first // 16 : (first + local) // 16, :, :]
                    codes[:, expert, matrix].copy_(packed.reshape(slots, 2, tiles, 32))
                    middle[expert, matrix].copy_(suh[first : first + local])
                    outputs[expert].copy_(svh)
    config = TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "lut_e4m3",
            "rate": {"granularity": "uniform"},
            "scale": {
                name: {"vectors": "per_expert", "gains": "none"}
                for name in ("input_scales", "intermediate_scales", "output_scales")
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        }
    )
    source = TrellisSource(
        config=config,
        uniform_bits=2,
        extent=TrellisExtent(
            global_intermediate_size=intermediate_size,
            first_slot=first // 32,
            slot_count=slots,
        ),
    )
    weights = TrellisWeights(
        codes=codes.view(torch.uint8).reshape(slots, -1),
        rate=torch.tensor(0x22, dtype=torch.uint8, device="cpu"),
        input_scales=ScaleFactors(inputs),
        intermediate_scales=ScaleFactors(middle),
        output_scales=ScaleFactors(outputs),
    )
    return source, weights
