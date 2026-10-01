"""Tensor sources for loading compressed FP4 expert weights.

The caller owns tensor names, file handles and model configuration. A packed
weight view permits TP slicing before materializing the weight bytes on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PackedTensorView(Protocol):
    """CPU tensor slicing interface provided by safetensors and tensor stores."""

    def get_shape(self) -> list[int]: ...

    def get_dtype(self) -> str: ...

    def __getitem__(self, extent: tuple[slice, slice]) -> torch.Tensor: ...


@dataclass(frozen=True)
class CsfMatrix:
    """One projection's lazy FP4 bytes and resident CPU scale components.

    NVFP4 also supplies the original FP32 scalar weight and activation calibration.
    MXFP4 leaves those fields unset. Keep the backing tensor store open until the
    weight-loading call returns.
    """

    weight: PackedTensorView
    fixed: torch.Tensor
    exceptions: torch.Tensor
    global_scale: torch.Tensor | None = None
    input_scale: torch.Tensor | None = None


def tp_extent(intermediate_size: int, tp_rank: int, tp_size: int, alignment: int):
    """Validate an equal TP partition and return its intermediate-axis bounds."""
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or intermediate_size <= 0
        or intermediate_size % (alignment * tp_size)
    ):
        raise ValueError(
            f"CSF requires a valid TP rank and extents aligned to {alignment} "
            "intermediate channels"
        )
    local = intermediate_size // tp_size
    return tp_rank * local, (tp_rank + 1) * local
