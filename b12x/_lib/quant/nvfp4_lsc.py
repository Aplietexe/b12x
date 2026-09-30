"""Routed lossless NVFP4 scale expansion into native F8_128x4 storage.

One grid expands gate/up and down scales. Each CTA owns a 128-row output
region, writes its fixed stream, and applies its prepartitioned exceptions
after a barrier. Output scratch is caller-owned and may be shared only by
serialized layer execution on one CUDA stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


@dataclass(frozen=True)
class Nvfp4LscBatch:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    task_offsets: torch.Tensor
    rows: int
    columns: int
    codec: int = 0

    @property
    def num_experts(self):
        return int(self.fixed.shape[0])

    @property
    def geometry(self):
        return self.rows, self.columns, self.codec

    def validate(self):
        if self.rows <= 0 or self.rows % 128 or self.columns <= 0 or self.columns % 8:
            raise ValueError("NVFP4-LSC requires 128-row and eight-column alignment")
        if self.rows * self.columns > (1 << (19 if self.codec == 2 else 24)):
            raise ValueError("NVFP4-LSC position field is too small for the matrix")
        if self.codec not in (0, 1, 2):
            raise ValueError("Unknown NVFP4-LSC scale codec")
        expected = (self.num_experts, self.rows // 16, 16 * (1 + self.columns // 2))
        if self.fixed.dtype != torch.uint8 or tuple(self.fixed.shape) != expected:
            raise ValueError("NVFP4-LSC fixed stream geometry or dtype mismatch")
        width = 3 if self.codec == 2 else 4
        if (
            self.exceptions.dtype != torch.uint8
            or self.exceptions.ndim != 1
            or self.exceptions.numel() % width
        ):
            raise ValueError("NVFP4-LSC exception stream extent or dtype mismatch")
        if self.task_offsets.dtype != torch.int64 or tuple(self.task_offsets.shape) != (
            self.num_experts,
            self.rows // 128 + 1,
        ):
            raise ValueError(
                "NVFP4-LSC requires expert-major 128-row exception partitions"
            )
        tensors = (self.fixed, self.exceptions, self.task_offsets)
        if self.fixed.device.type != "cuda" or any(
            t.device != self.fixed.device or not t.is_contiguous() for t in tensors
        ):
            raise ValueError(
                "NVFP4-LSC runtime tensors must be contiguous on one CUDA device"
            )


def make_nvfp4_lsc_batch(
    fixed_planes, exception_planes, *, rows, columns, device, codec=0
):
    """Upload compressed CPU planes and build immutable exception partitions."""
    if not fixed_planes or len(fixed_planes) != len(exception_planes):
        raise ValueError("NVFP4-LSC component lists must be nonempty and equally sized")
    if rows <= 0 or rows % 128 or columns <= 0 or columns % 8 or codec not in (0, 1, 2):
        raise ValueError("Unsupported NVFP4-LSC geometry or codec")
    position_bits = 19 if codec == 2 else 24
    if rows * columns > 1 << position_bits:
        raise ValueError("NVFP4-LSC position field is too small for the matrix")
    fixed_list, exception_list, partitions = [], [], []
    cursor = 0
    for fixed, exceptions in zip(fixed_planes, exception_planes, strict=True):
        f = np.asarray(fixed, dtype=np.uint8).reshape(
            rows // 16, 16 * (1 + columns // 2)
        )
        e = np.asarray(exceptions).view(np.uint8).reshape(-1)
        width = 3 if codec == 2 else 4
        if len(e) % width:
            raise ValueError("Invalid NVFP4-LSC exception record length")
        if width == 4:
            words = e.copy().view("<u4")
        else:
            b = e.reshape(-1, 3).astype(np.uint32)
            words = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        positions = words & ((1 << position_bits) - 1)
        if len(words) and (
            positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError("NVFP4-LSC exceptions must be unique, sorted and in range")
        bounds = np.arange(rows // 128 + 1, dtype=np.int64) * (128 * columns)
        partitions.append(np.searchsorted(positions, bounds).astype(np.int64) + cursor)
        cursor += len(words)
        fixed_list.append(f)
        exception_list.append(e)
    result = Nvfp4LscBatch(
        torch.from_numpy(np.stack(fixed_list)).to(device),
        torch.from_numpy(np.concatenate(exception_list)).to(device),
        torch.from_numpy(np.stack(partitions)).to(device),
        int(rows),
        int(columns),
        int(codec),
    )
    result.validate()
    return result


class _Plane:
    def __init__(self, geometry):
        self.rows, self.columns, self.codec = map(int, geometry)
        self.tasks = self.rows // 128
        self.tile_bytes = 16 * (1 + self.columns // 2)

    @cute.jit
    def tensors(self, fixed, exceptions, offsets, output, experts, exception_bytes):
        return (
            cute.make_tensor(
                fixed,
                cute.make_layout(
                    (Int64(experts) * Int64(self.rows // 16 * self.tile_bytes),)
                ),
            ),
            cute.make_tensor(exceptions, cute.make_layout((exception_bytes,))),
            cute.make_tensor(
                offsets, cute.make_layout((Int64(experts) * Int64(self.tasks + 1),))
            ),
            cute.make_tensor(
                output,
                cute.make_layout((Int64(experts) * Int64(self.rows * self.columns),)),
            ),
        )

    @cute.jit
    def output_offset(self, row, column):
        return (
            (
                ((row // Int32(128)) * Int32(self.columns // 4) + column // Int32(4))
                * Int32(32)
                + row % Int32(32)
            )
            * Int32(16)
            + ((row % Int32(128)) // Int32(32)) * Int32(4)
            + column % Int32(4)
        )

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        fixed, exceptions, offsets, output = tensors
        fixed16 = cute.recast_tensor(fixed, cutlass.Uint16)
        output32 = cute.recast_tensor(output, cutlass.Uint32)
        word = tid
        while word < Int32(128 * self.columns // 4):
            local = word % Int32(128)
            row = task * Int32(128) + (local % Int32(4)) * Int32(32) + local // Int32(4)
            column = (word // Int32(128)) * Int32(4)
            tile = (
                Int64(expert) * Int64(self.rows // 16) + Int64(row // Int32(16))
            ) * Int64(self.tile_bytes)
            rr = row % Int32(16)
            base = fixed[tile + Int64(rr)].to(Uint32)
            values = Uint32(0)
            if cutlass.const_expr(self.codec == 0):
                address = tile + Int64(
                    16 + rr * Int32(self.columns // 2) + column // Int32(2)
                )
                packed = fixed16[address >> Int64(1)].to(Uint32)
                for lane in cutlass.range_constexpr(4):
                    value = base + ((packed >> Uint32(lane * 4)) & Uint32(15))
                    values |= value << Uint32(lane * 8)
            else:
                selector = fixed[
                    tile
                    + Int64(16 + rr * Int32(self.columns // 8) + column // Int32(8))
                ].to(Uint32)
                mantissa_address = tile + Int64(
                    16
                    + 16 * (self.columns // 8)
                    + rr * Int32(3 * self.columns // 8)
                    + (column // Int32(8)) * Int32(3)
                )
                mantissa = (
                    fixed[mantissa_address].to(Uint32)
                    | (fixed[mantissa_address + Int64(1)].to(Uint32) << Uint32(8))
                    | (fixed[mantissa_address + Int64(2)].to(Uint32) << Uint32(16))
                )
                for lane in cutlass.range_constexpr(4):
                    cc = Uint32(column % Int32(8) + Int32(lane))
                    high = base + ((selector >> cc) & Uint32(1))
                    value = (high << Uint32(3)) | (
                        (mantissa >> (cc * Uint32(3))) & Uint32(7)
                    )
                    values |= value << Uint32(8 * lane)
            destination = Int64(expert) * Int64(self.rows * self.columns) + Int64(
                self.output_offset(row, column)
            )
            output32[destination >> Int64(2)] = values
            word += Int32(256)
        cute.arch.sync_threads()
        partition = Int64(expert) * Int64(self.tasks + 1) + Int64(task)
        entry = offsets[partition] + Int64(tid)
        end = offsets[partition + Int64(1)]
        while entry < end:
            if cutlass.const_expr(self.codec == 2):
                address = entry * Int64(3)
                packed = (
                    exceptions[address].to(Uint32)
                    | (exceptions[address + Int64(1)].to(Uint32) << Uint32(8))
                    | (exceptions[address + Int64(2)].to(Uint32) << Uint32(16))
                )
                position = packed & Uint32((1 << 19) - 1)
                value = packed >> Uint32(19)
            else:
                words = cute.recast_tensor(exceptions, cutlass.Uint32)
                packed = words[entry]
                position = packed & Uint32(0xFFFFFF)
                value = packed >> Uint32(24)
            row = Int32(position // Uint32(self.columns))
            column = Int32(position % Uint32(self.columns))
            destination = Int64(expert) * Int64(self.rows * self.columns) + Int64(
                self.output_offset(row, column)
            )
            if cutlass.const_expr(self.codec != 0):
                value = (value << Uint32(3)) | (
                    output[destination].to(Uint32) & Uint32(7)
                )
            output[destination] = value.to(cutlass.Uint8)
            entry += Int64(256)


class _Pair:
    def __init__(self, first, second):
        self.first, self.second = _Plane(first), _Plane(second)

    @cute.jit
    def __call__(
        self,
        f13: cute.Pointer,
        e13: cute.Pointer,
        p13: cute.Pointer,
        o13: cute.Pointer,
        f2: cute.Pointer,
        e2: cute.Pointer,
        p2: cute.Pointer,
        o2: cute.Pointer,
        ids_ptr: cute.Pointer,
        experts: Int32,
        bytes13: Int64,
        bytes2: Int64,
        capacity: Int32,
        mode: Int32,
        stream: cuda.CUstream,
    ):
        first = self.first.tensors(f13, e13, p13, o13, experts, bytes13)
        second = self.second.tensors(f2, e2, p2, o2, experts, bytes2)
        ids = cute.make_tensor(ids_ptr, cute.make_layout((capacity,)))
        self.kernel(first, second, ids, experts, capacity, mode).launch(
            grid=(capacity * Int32(self.first.tasks + self.second.tasks), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, first, second, ids, experts: Int32, capacity: Int32, mode: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        slot = Int32(block) // Int32(self.first.tasks + self.second.tasks)
        task = Int32(block) % Int32(self.first.tasks + self.second.tasks)
        expert = slot
        # mode 0: potentially repeated routes; mode 1: unique IDs;
        # mode 2: expert counts; mode 3: every expert, no route read.
        if mode != Int32(3):
            raw = ids[slot].to(Int64)
            if mode == Int32(2):
                if raw <= Int64(0):
                    expert = Int32(-1)
            else:
                expert = Int32(-1)
                if raw >= Int64(0) and raw < Int64(experts):
                    expert = raw.to(Int32)
                if mode == Int32(0):
                    prior = Int32(0)
                    while prior < slot and expert >= Int32(0):
                        if ids[prior].to(Int64) == raw:
                            expert = Int32(-1)
                        prior += Int32(1)
        if expert >= Int32(0) and expert < experts:
            if task < Int32(self.first.tasks):
                self.first.decode(first, expert, task, Int32(tid))
            else:
                self.second.decode(
                    second, expert, task - Int32(self.first.tasks), Int32(tid)
                )


@program_cache
def compile_nvfp4_lsc_pair(first, second, ids64=False):
    launch = _Pair(first, second)
    key = (first, second, bool(ids64))
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    plane = (
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
    )
    return b12x_compile(
        launch,
        *plane,
        *plane,
        make_ptr(
            cutlass.Int64 if ids64 else cutlass.Int32,
            8 if ids64 else 4,
            cute.AddressSpace.gmem,
            assumed_align=8 if ids64 else 4,
        ),
        1,
        1,
        1,
        1,
        0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.nvfp4_lsc_pair", 1, key),
    )


def decode_nvfp4_lsc_pair(first, second, ids, out13, out2, *, mode=0, program=None):
    """Launch the precompiled pair into caller-owned native NVFP4 scale grids."""
    if first.num_experts != second.num_experts or mode not in (0, 1, 2, 3):
        raise ValueError("NVFP4-LSC expert counts or routing mode disagree")
    if (
        ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.device != first.fixed.device
    ):
        raise ValueError("NVFP4-LSC routes must be contiguous CUDA int32/int64")
    for batch, out in ((first, out13), (second, out2)):
        if (
            out.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or tuple(out.shape) != (batch.num_experts, batch.rows, batch.columns)
            or not out.is_contiguous()
            or out.device != batch.fixed.device
        ):
            raise ValueError("NVFP4-LSC output must match the native scale geometry")
    capacity = first.num_experts if mode == 3 else ids.numel()
    if mode == 2 and capacity != first.num_experts:
        raise ValueError("NVFP4-LSC count routing requires one count per expert")
    if not capacity:
        return
    if program is None:
        program = compile_nvfp4_lsc_pair(
            first.geometry, second.geometry, ids.dtype == torch.int64
        )

    def device_ptr(t, dtype, align):
        return make_ptr(
            dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    args = []
    for batch, out in ((first, out13), (second, out2)):
        args.extend(
            (
                device_ptr(batch.fixed, cutlass.Uint8, 16),
                device_ptr(batch.exceptions, cutlass.Uint8, 16),
                device_ptr(batch.task_offsets, cutlass.Int64, 8),
                device_ptr(out, cutlass.Uint8, 16),
            )
        )
    program(
        *args,
        device_ptr(
            ids,
            cutlass.Int64 if ids.dtype == torch.int64 else cutlass.Int32,
            8 if ids.dtype == torch.int64 else 4,
        ),
        first.num_experts,
        first.exceptions.numel(),
        second.exceptions.numel(),
        capacity,
        mode,
        current_cuda_stream(),
    )


@dataclass(frozen=True)
class Nvfp4LscDecoder:
    """Compressed expert planes and retained int32/int64 routing programs."""

    first: Nvfp4LscBatch
    second: Nvfp4LscBatch
    programs: tuple

    @classmethod
    def prepare(cls, first, second, out13, out2):
        for plane, output in ((first, out13), (second, out2)):
            plane.validate()
            expected = (plane.num_experts, plane.rows, plane.columns)
            if (
                tuple(output.shape) != expected
                or output.dtype != torch.float8_e4m3fn
                or output.device != plane.fixed.device
                or not output.is_contiguous()
            ):
                raise ValueError(
                    "NVFP4-LSC scratch must match native E4M3 scale storage"
                )
        if first.num_experts != second.num_experts:
            raise ValueError("NVFP4-LSC projections must have equal expert counts")
        programs = tuple(
            compile_nvfp4_lsc_pair(first.geometry, second.geometry, ids64)
            for ids64 in (False, True)
        )
        return cls(first, second, programs)

    def decode(self, ids, out13, out2):
        # Sparse calls expand only selected experts. Above one route per
        # expert on average, a full grid bounds duplicate-search work.
        mode = 0 if ids.numel() < self.first.num_experts else 3
        decode_nvfp4_lsc_pair(
            self.first,
            self.second,
            ids,
            out13,
            out2,
            mode=mode,
            program=self.programs[int(ids.dtype == torch.int64)],
        )
