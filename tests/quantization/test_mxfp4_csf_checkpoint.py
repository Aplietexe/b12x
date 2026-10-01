"""Byte-level checks for slicing compressed UE8M0 scales across TP ranks."""

import numpy as np
import pytest
import torch

from b12x.moe.checkpoints.mxfp4_csf import slice_scale_plane

from ..conftest import require_b12x
from .csf_fixtures import matrix


def decode(fixed, exceptions, rows, columns):
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )[:, :columns]
    result = (stream[:, :16].reshape(rows, 1) + bits).astype(np.uint8)
    words = exceptions.numpy()
    result.flat[words & 0xFFFFFF] = words >> 24
    return result


@pytest.mark.parametrize("columns", [9, 18, 36, 72, 160])
@pytest.mark.parametrize("row_slice", [(0, 64), (16, 48), (48, 64)])
def test_slices_preserve_all_scale_bytes(columns, row_slice):
    rng = np.random.default_rng(78)
    rows = 64
    bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
    bases = rng.integers(0, 254, (rows // 16, 16), dtype=np.uint8)
    fixed = torch.from_numpy(
        np.concatenate(
            (
                bases,
                np.packbits(bits, axis=1, bitorder="little").reshape(rows // 16, -1),
            ),
            1,
        )
    )
    positions = np.unique(rng.integers(0, rows * columns, 50)).astype(np.uint32)
    values = rng.integers(0, 256, len(positions), dtype=np.uint32)
    exceptions = torch.from_numpy(positions | values << 24)
    reference = decode(fixed, exceptions, rows, columns)
    for c0, c1 in ((0, columns), (1, columns), (columns // 2, columns)):
        sliced = slice_scale_plane(
            fixed, exceptions, rows, columns, row_slice, (c0, c1)
        )
        actual = decode(*sliced, row_slice[1] - row_slice[0], c1 - c0)
        assert np.array_equal(actual, reference[row_slice[0] : row_slice[1], c0:c1])


def test_rejects_unaligned_rows_and_bad_padding():
    fixed = torch.zeros((4, 48), dtype=torch.uint8)
    exceptions = torch.empty(0, dtype=torch.uint32)
    with pytest.raises(ValueError, match="16-row"):
        slice_scale_plane(fixed, exceptions, 64, 9, (1, 17), (0, 9))
    fixed[0, 17] = 128
    with pytest.raises(ValueError, match="unused selector"):
        slice_scale_plane(fixed, exceptions, 64, 9, (0, 64), (0, 9))


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_sources_preserve_projection_order_and_tp_bytes(rank):
    """An arbitrary geometry loads without any model identity or checkpoint path."""
    from b12x._lib.quant.x4t_scales import decode_x4t_scales
    from b12x.moe.checkpoints.mxfp4_csf import load_mxfp4_csf_weights

    device = require_b12x()
    experts, hidden, intermediate, local = 3, 256, 192, 96
    sources, scales = [], []
    for expert in range(experts):
        pairs = [
            matrix(r, c, group_size=32, seed=expert * 17 + projection)
            for projection, (r, c) in enumerate(
                ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
            )
        ]
        sources.append(tuple(pair[0] for pair in pairs))
        scales.append(tuple(pair[1] for pair in pairs))
    scratch13 = torch.empty(
        (experts, hidden // 32, 2 * local), dtype=torch.uint8, device=device
    )
    scratch2 = torch.empty(
        (experts, local // 32, hidden), dtype=torch.uint8, device=device
    )
    weights = load_mxfp4_csf_weights(
        iter(sources),
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tp_rank=rank,
        tp_size=2,
        device=device,
        w13_scale_scratch=scratch13,
        w2_scale_scratch=scratch2,
    )
    first, last = rank * local, (rank + 1) * local
    expected13 = torch.stack(
        [
            torch.cat((gate.weight[first:last, :], up.weight[first:last, :]))
            for gate, up, _ in sources
        ]
    )
    expected2 = torch.stack(
        [down.weight[:, first // 2 : last // 2] for _, _, down in sources]
    )
    assert torch.equal(weights.w13.cpu(), expected13)
    assert torch.equal(weights.w2.cpu(), expected2)
    assert (
        weights.w13_scale_scratch is scratch13 and weights.w2_scale_scratch is scratch2
    )
    ids = torch.arange(experts, device=device, dtype=torch.int32)
    for batch, expected in (
        (
            weights.w13_scales,
            torch.stack(
                [torch.cat((s[0][first:last], s[1][first:last])) for s in scales]
            ),
        ),
        (
            weights.w2_scales,
            torch.stack([s[2][:, first // 32 : last // 32] for s in scales]),
        ),
    ):
        output = torch.empty_like(expected, device=device)
        decode_x4t_scales(batch, ids, output)
        assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("count", [0, 2])
def test_expert_inventory_must_match_declared_count(count):
    from b12x.moe.checkpoints.mxfp4_csf import load_mxfp4_csf_weights

    source, _ = matrix(128, 128, group_size=32, seed=0)
    with pytest.raises(ValueError, match="zip"):
        load_mxfp4_csf_weights(
            [(source, source, source)] * count,
            num_experts=1,
            hidden_size=128,
            intermediate_size=128,
            tp_rank=0,
            tp_size=1,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )
