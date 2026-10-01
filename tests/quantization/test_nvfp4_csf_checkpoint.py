"""Tensor-driven NVFP4 loading preserves TP bytes and FP32 calibration."""

from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x
from .csf_fixtures import matrix


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_sources_preserve_tp_bytes_and_fp32_with_bf16_default(rank):
    from b12x._lib.quant.nvfp4_csf import decode_nvfp4_csf_pair
    from b12x.moe.checkpoints.nvfp4_csf import load_nvfp4_csf_weights

    device = require_b12x()
    experts, hidden, intermediate, local = 3, 256, 256, 128
    calibration = torch.linspace(0.013123, 0.056789, experts, dtype=torch.float32)
    gate_input = calibration * 19.321
    up_input = calibration * 20.123
    down_input = calibration * 31.345
    sources, scales = [], []
    for expert in range(experts):
        pairs = [
            matrix(r, c, group_size=16, seed=expert * 17 + projection)
            for projection, (r, c) in enumerate(
                ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
            )
        ]
        sources.append(
            tuple(
                replace(p[0], global_scale=calibration[expert], input_scale=a[expert])
                for p, a in zip(pairs, (up_input, gate_input, down_input), strict=True)
            )
        )
        scales.append(tuple(pair[1] for pair in pairs))
    scratch13 = torch.empty(
        (experts, 2 * local, hidden // 16), device=device, dtype=torch.float8_e4m3fn
    )
    scratch2 = torch.empty(
        (experts, hidden, local // 16), device=device, dtype=torch.float8_e4m3fn
    )
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        weights = load_nvfp4_csf_weights(
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
    finally:
        torch.set_default_dtype(default)
    for name, expected in (
        ("w13_global_scales", calibration.to(device)),
        ("w2_global_scales", calibration.to(device)),
        ("input_scale", up_input.to(device).reciprocal()),
        ("intermediate_scale", down_input.to(device).reciprocal()),
    ):
        actual = getattr(weights.packed, name)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    first, last = rank * local, (rank + 1) * local
    assert torch.equal(
        weights.packed.w13.cpu(),
        torch.stack(
            [
                torch.cat((up.weight[first:last, :], gate.weight[first:last, :]))
                for up, gate, _ in sources
            ]
        ),
    )
    assert torch.equal(
        weights.packed.w2.cpu(),
        torch.stack([down.weight[:, first // 2 : last // 2] for _, _, down in sources]),
    )
    assert weights.packed.w13_block_scales is scratch13
    assert weights.packed.w2_block_scales is scratch2
    expected13 = torch.stack(
        [torch.cat((s[0][first:last], s[1][first:last])) for s in scales]
    )
    expected2 = torch.stack([s[2][:, first // 16 : last // 16] for s in scales])
    out13, out2 = (
        torch.empty_like(expected13, device=device),
        torch.empty_like(expected2, device=device),
    )
    ids = torch.arange(experts, device=device, dtype=torch.int32)
    decode_nvfp4_csf_pair(weights.w13_scales, weights.w2_scales, ids, out13, out2)
    for actual, expected in ((out13, expected13), (out2, expected2)):
        e, r, c = expected.shape
        packed = expected.reshape(e, r // 128, 4, 32, c // 4, 4)
        packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous().reshape_as(expected)
        assert torch.equal(actual.cpu(), packed)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.tensor(1.0, dtype=torch.bfloat16),
        torch.tensor(float("nan")),
        torch.tensor(0.0),
    ],
)
def test_missing_or_invalid_calibration_is_rejected(bad):
    from b12x.moe.checkpoints.nvfp4_csf import load_nvfp4_csf_weights

    source, _ = matrix(128, 128, group_size=16, seed=0)
    source = replace(source, global_scale=bad, input_scale=torch.tensor(1.0))
    with pytest.raises(ValueError, match="global_scale"):
        load_nvfp4_csf_weights(
            [(source, source, source)],
            num_experts=1,
            hidden_size=128,
            intermediate_size=128,
            tp_rank=0,
            tp_size=1,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )
