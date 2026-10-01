"""Model-independent TP loading of compressed MXFP4 expert tensors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x.moe.checkpoints.csf import CsfMatrix, tp_extent
from b12x.moe.fused_moe.weights import Mxfp4CsfWeights

POSITION_MASK = (1 << 24) - 1


def slice_scale_plane(fixed, exceptions, rows, columns, row_slice, column_slice):
    """Slice compressed bytes without floating-point reconstruction or fitting."""
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("MXFP4-CSF row slices must contain complete 16-row slabs")
    if not 0 <= c0 < c1 <= columns:
        raise ValueError("MXFP4-CSF column slice is outside the scale plane")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("MXFP4-CSF requires uint8 fixed bytes and uint32 exceptions")
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bases = stream[:, :16]
    if (bases > 254).any():
        raise ValueError("MXFP4-CSF palette bases must be in 0..254")
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )
    if bits[:, columns:].any():
        raise ValueError("MXFP4-CSF unused selector bits must be zero")
    selected = np.packbits(bits[r0:r1, c0:c1], axis=1, bitorder="little")
    result = np.concatenate(
        (bases[r0 // 16 : r1 // 16], selected.reshape((r1 - r0) // 16, -1)), 1
    )
    words = exceptions.numpy().reshape(-1)
    positions = words & POSITION_MASK
    if len(words) and (
        positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError(
            "MXFP4-CSF exception positions must be unique, sorted and in range"
        )
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
    words = (words[keep] & np.uint32(0xFF000000)) | positions
    return torch.from_numpy(result.copy()), torch.from_numpy(words.astype(np.uint32))


def load_mxfp4_csf_weights(
    experts: Iterable[tuple[CsfMatrix, CsfMatrix, CsfMatrix]],
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
    """Slice and upload gate/up/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    """
    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 64:
        raise ValueError("MXFP4-CSF requires experts and 64-aligned hidden channels")
    first, last = tp_extent(intermediate_size, tp_rank, tp_size, 32)
    local = last - first
    w13 = torch.empty(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = torch.empty(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []
    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        f13, e13 = [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() not in ("I8", "U8"):
                raise ValueError(
                    f"MXFP4-CSF nibble geometry/dtype mismatch: expert={expert}, projection={matrix}"
                )
            if matrix < 2:
                w13[expert, matrix * local : (matrix + 1) * local].copy_(
                    view[first:last, :].view(torch.uint8)
                )
                rows, columns = intermediate_size, hidden_size // 32
                row_slice, column_slice = (first, last), (0, columns)
            else:
                w2[expert].copy_(view[:, first // 2 : last // 2].view(torch.uint8))
                rows, columns = hidden_size, intermediate_size // 32
                row_slice, column_slice = (0, rows), (first // 32, last // 32)
            fixed, exceptions = slice_scale_plane(
                projection.fixed,
                projection.exceptions,
                rows,
                columns,
                row_slice,
                column_slice,
            )
            if matrix < 2:
                f13.append(fixed)
                if matrix:
                    words = exceptions.numpy().copy()
                    words += np.uint32(local * columns)
                    exceptions = torch.from_numpy(words)
                e13.append(exceptions)
            else:
                fixed2.append(fixed)
                exceptions2.append(exceptions)
        fixed13.append(torch.cat(f13))
        exceptions13.append(torch.cat(e13))
    batch13 = make_x4t_scale_batch(
        fixed13,
        exceptions13,
        rows=2 * local,
        columns=hidden_size // 32,
        device=device,
        exception_task_rows=64,
    )
    batch2 = make_x4t_scale_batch(
        fixed2,
        exceptions2,
        rows=hidden_size,
        columns=local // 32,
        device=device,
        exception_task_rows=64,
    )
    return Mxfp4CsfWeights(
        w13=w13.to(device),
        w2=w2.to(device),
        w13_scales=batch13,
        w2_scales=batch2,
        w13_scale_scratch=w13_scale_scratch,
        w2_scale_scratch=w2_scale_scratch,
    )
