"""NVFP4 checkpoint calibration retains FP32 under BF16 model loading."""

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reader_preserves_fp32_calibration_with_bf16_default(monkeypatch):
    from b12x.moe.checkpoints import nvfp4_csf as reader

    device = require_b12x()
    calibration = torch.linspace(0.013123, 0.056789, 288, dtype=torch.float32)
    gate_input = calibration * 19.321
    up_input = calibration * 20.123
    down_input = calibration * 31.345
    gate_weight = torch.zeros((2048, 2048), dtype=torch.uint8)
    down_weight = torch.zeros((4096, 1024), dtype=torch.uint8)
    fixed13 = torch.zeros((128, 16 * 129), dtype=torch.uint8)
    fixed2 = torch.zeros((256, 16 * 65), dtype=torch.uint8)
    exceptions = torch.empty(0, dtype=torch.uint32)

    class Slice:
        def __init__(self, tensor):
            self.tensor = tensor

        def get_shape(self):
            return list(self.tensor.shape)

        def get_dtype(self):
            return "U8"

        def __getitem__(self, item):
            return self.tensor[item]

    class Shard:
        def get_slice(self, name):
            return Slice(down_weight if ".down_proj." in name else gate_weight)

        def get_tensor(self, name):
            if name.endswith(".nvfp4_csf_fixed"):
                return fixed2 if ".down_proj." in name else fixed13
            if name.endswith(".nvfp4_csf_exceptions"):
                return exceptions
            expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            if name.endswith(".weight_scale_2"):
                return calibration[expert]
            assert name.endswith(".input_scale")
            values = (
                down_input
                if ".down_proj." in name
                else up_input
                if ".up_proj." in name
                else gate_input
            )
            return values[expert]

    monkeypatch.setattr(
        reader,
        "checkpoint_contract",
        lambda _: {
            "family": "glm53_nvfp4",
            "source_names": defaultdict(lambda: "calibration.safetensors"),
        },
    )
    monkeypatch.setattr(reader, "safe_open", lambda *a, **k: nullcontext(Shard()))
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        weights = reader.read_nvfp4_csf_layer(
            Path("/synthetic-nvfp4-csf"),
            3,
            num_experts=288,
            hidden_size=4096,
            intermediate_size=2048,
            tp_rank=7,
            tp_size=16,
            device=device,
            w13_scale_scratch=torch.empty(
                (288, 256, 256), device=device, dtype=torch.float8_e4m3fn
            ),
            w2_scale_scratch=torch.empty(
                (288, 4096, 8), device=device, dtype=torch.float8_e4m3fn
            ),
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
