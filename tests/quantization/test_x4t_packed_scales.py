from __future__ import annotations

import numpy as np
import pytest
import torch

from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x._lib.quant.x4t_packed_scales import (
    decode_x4t_packed_scales,
    _compiled_packed_scale,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.prepare import _pack_e8m0_k32_scales
from ..conftest import require_b12x


def _batch(rows, columns, rotation, task_rows=64):
    device = require_b12x()
    fixed, exceptions, logical = [], [], []
    for expert in range(4):
        bits = (np.arange(rows * columns).reshape(rows, columns) + expert) % 2
        selectors = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
        base = 120 + expert
        values = (bits + base).astype(np.uint8)
        positions = np.unique(
            [
                0,
                63 * columns,
                64 * columns - 1,
                64 * columns,
                (rows // 2 + 1) * columns,
                rows * columns - 1,
            ]
        )
        overrides = np.array([0, 246, 247, 248, 254, 255], dtype=np.uint32)[
            -len(positions) :
        ]
        values.flat[positions] = overrides
        words = positions.astype(np.uint32) | (overrides << 24)
        bases = np.full((rows // 16, 16), base, dtype=np.uint8)
        stream = np.concatenate((bases, selectors.reshape(rows // 16, -1)), axis=1)
        fixed.append(torch.from_numpy(stream))
        exceptions.append(torch.from_numpy(words))
        logical.append(torch.from_numpy(values))
    batch = make_x4t_scale_batch(
        fixed,
        exceptions,
        rows=rows,
        columns=columns,
        device=device,
        exception_task_rows=task_rows,
        exception_row_rotation=rotation,
    )
    return batch, torch.stack(logical).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "rows,columns,rotation",
    [
        (128, 1, 0),
        (128, 18, 64),
        (1152, 160, 0),
        (5120, 18, 0),
        (2304, 160, 1152),
        (5120, 36, 0),
        (4608, 160, 0),
        (5120, 72, 0),
    ],
)
def test_packed_scale_exact_boundaries_duplicates_and_dynamic_graph(
    rows, columns, rotation
):
    batch, logical = _batch(rows, columns, rotation)
    output = torch.full(
        (4, columns, rows), 0xD6, dtype=torch.uint8, device=logical.device
    )
    ids = torch.tensor([3, 1, 3, -1, 4], dtype=torch.int32, device=logical.device)
    decode_x4t_packed_scales(batch, ids, output)
    reference = _pack_e8m0_k32_scales(
        logical, size_k=columns * 32, size_n=rows, row_rotation=rotation
    ).view(torch.uint8)
    assert torch.equal(output[[3, 1]], reference[[3, 1]])
    assert bool((output[[0, 2]] == 0xD6).all())
    misses = _compiled_packed_scale.cache_info().misses
    with kernel_resolution_guard("packed-scale graph qualification"):
        for count in (1, 3, 5):
            decode_x4t_packed_scales(batch, ids[:count], output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_x4t_packed_scales(batch, ids, output)
        ids.copy_(torch.tensor([2, 0, 2, -1, 7], dtype=torch.int32, device=ids.device))
        output.fill_(0xD6)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
    assert _compiled_packed_scale.cache_info().misses == misses
    assert torch.equal(output[[2, 0]], reference[[2, 0]])
    assert bool((output[[1, 3]] == 0xD6).all())
    graph.reset()
    # The exact-byte mode preserves the full UE8M0 alphabet. The serving clamp
    # is a distinct operation shared with native W4A16 preparation.
    ids_unique = torch.arange(4, dtype=torch.int32, device=ids.device)
    decode_x4t_packed_scales(
        batch, ids_unique, output, clamp_e8m0_bf16=False, expert_ids_unique=True
    )
    from b12x.moe._shared.kernels.w4a16.prepare import _scale_perms

    permutation = _scale_perms()[int(columns == 1)]
    rotated = torch.roll(logical, -rotation, dims=1).transpose(1, 2).contiguous()
    exact = rotated.reshape(-1, len(permutation))[:, permutation]
    exact = exact.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape_as(output)
    assert torch.equal(output, exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_scale_rejects_misaligned_exception_partition():
    batch, logical = _batch(128, 18, 0, task_rows=32)
    ids = torch.arange(4, dtype=torch.int32, device=logical.device)
    output = torch.empty((4, 18, 128), dtype=torch.uint8, device=logical.device)
    with pytest.raises(ValueError, match="64 rows"):
        decode_x4t_packed_scales(batch, ids, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_counts_retained_program_and_poisoned_graph():
    batch, logical = _batch(1152, 160, 0)
    output = torch.full((4, 160, 1152), 0xD6, dtype=torch.uint8, device=logical.device)
    counts = torch.tensor([3, 0, 8, 0], dtype=torch.int32, device=logical.device)
    program = _compiled_packed_scale(1152, 160, 64, 0, True, False, True)
    reference = _pack_e8m0_k32_scales(logical, size_k=5120, size_n=1152).view(torch.uint8)
    decode_x4t_packed_scales(batch, counts, output, expert_counts=True, program=program)
    assert torch.equal(output[[0, 2]], reference[[0, 2]])
    _compiled_packed_scale.cache_clear()
    with kernel_resolution_guard("retained X4T counts decoder"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_x4t_packed_scales(batch, counts, output, expert_counts=True, program=program)
        for active in ([0, 4, 0, 10], [0, 0, 0, 0], [1, 0, 4, 2]):
            counts.copy_(torch.tensor(active, dtype=torch.int32, device=counts.device))
            output.fill_(0xD6)
            graph.replay()
            torch.cuda.synchronize()
            for expert, count in enumerate(active):
                if count:
                    assert torch.equal(output[expert], reference[expert])
                else:
                    assert bool((output[expert] == 0xD6).all())
    graph.reset()
