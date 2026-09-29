"""Expand an exact-MXFP4 scale plane into the packed W4A16 layout.

Each CTA owns an aligned region of packed rows. It writes the adjacent-pair
palette first, then applies that region's sparse exceptions after a barrier.
This permits one launch per plane, including non-byte-aligned selector rows.
Callers own the output scratch and build exception task offsets during loading.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.x4t_scales import X4TScaleBatch
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _PackedScaleDecode:
    def __init__(self, rows, columns, task_rows, rotation, clamp, unique, counts, ids64):
        self.rows = int(rows)
        self.columns = int(columns)
        self.task_rows = int(task_rows)
        self.tasks = self.rows // self.task_rows
        self.selector_bytes = (self.columns + 7) // 8
        self.tile_bytes = 16 * (1 + self.selector_bytes)
        self.rotation = int(rotation)
        self.clamp = bool(clamp)
        self.unique = bool(unique)
        self.counts = bool(counts)
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        fixed_ptr: cute.Pointer,
        exceptions_ptr: cute.Pointer,
        offsets_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        experts: Int32,
        exceptions_count: Int64,
        capacity: Int32,
        stream: cuda.CUstream,
    ):
        fixed = cute.make_tensor(
            fixed_ptr,
            cute.make_layout(
                (Int64(experts) * Int64(self.rows // 16 * self.tile_bytes),)
            ),
        )
        exceptions = cute.make_tensor(
            exceptions_ptr, cute.make_layout((exceptions_count,))
        )
        offsets = cute.make_tensor(
            offsets_ptr, cute.make_layout((Int64(experts) * Int64(self.tasks + 1),))
        )
        ids = cute.make_tensor(ids_ptr, cute.make_layout((capacity,)))
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((Int64(experts) * Int64(self.rows * self.columns),)),
        )
        self.kernel(fixed, exceptions, offsets, ids, output, experts, capacity).launch(
            grid=(capacity * Int32(self.tasks), 1, 1),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _clamp(self, value: Uint32) -> Uint32:
        if cutlass.const_expr(self.clamp):
            if value > Uint32(247):
                value = Uint32(247)
        return value

    @cute.kernel
    def kernel(
        self,
        fixed: cute.Tensor,
        exceptions: cute.Tensor,
        offsets: cute.Tensor,
        ids: cute.Tensor,
        output: cute.Tensor,
        experts: Int32,
        capacity: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        slot = Int32(block) // Int32(self.tasks)
        task = Int32(block) % Int32(self.tasks)
        raw_id = ids[slot].to(Int64)
        expert = Int32(-1)
        if raw_id >= Int64(0) and raw_id < Int64(experts):
            expert = raw_id.to(Int32)
        if cutlass.const_expr(self.counts):
            expert = Int32(-1)
            if ids[slot].to(Int32) > Int32(0):
                expert = slot
        if cutlass.const_expr(not self.unique and not self.counts):
            # All threads resolve the same immutable list. Repeated IDs must
            # have only one writer because exceptions follow the base stores.
            previous = Int32(0)
            while previous < slot and expert >= Int32(0):
                if ids[previous].to(Int64) == Int64(expert):
                    expert = Int32(-1)
                previous += Int32(1)
        if expert >= Int32(0) and expert < experts:
            output_u32 = cute.recast_tensor(output, cutlass.Uint32)
            row_base = task * Int32(self.task_rows)
            word = Int32(tid)
            while word < Int32(self.task_rows * self.columns // 4):
                element = word << Int32(2)
                column = element // Int32(self.task_rows)
                packed_in_task = element % Int32(self.task_rows)
                packed_values = Uint32(0)
                for lane in cutlass.range_constexpr(4):
                    packed_row = row_base + packed_in_task + Int32(lane)
                    low = (packed_row & Int32(63)) >> Int32(3)
                    high = packed_row & Int32(7)
                    swapped = (
                        (high & Int32(4))
                        | ((high & Int32(1)) << Int32(1))
                        | ((high >> Int32(1)) & Int32(1))
                    )
                    source_row = (packed_row & ~Int32(63)) + (swapped << Int32(3)) + low
                    if cutlass.const_expr(self.columns == 1):
                        # W4A16 uses a separate 32-row permutation when K has
                        # one scale group. Undo the final four-byte swap first.
                        p = packed_row & Int32(31)
                        p = (
                            (p & ~Int32(3))
                            | ((p & Int32(1)) << Int32(1))
                            | ((p >> Int32(1)) & Int32(1))
                        )
                        source_row = (
                            (packed_row & ~Int32(31))
                            + ((p >> Int32(3)) << Int32(1))
                            + (p & Int32(1))
                            + ((p & Int32(6)) << Int32(2))
                        )
                    if cutlass.const_expr(self.rotation):
                        source_row = (source_row + Int32(self.rotation)) % Int32(
                            self.rows
                        )
                    tile_base = (
                        Int64(expert) * Int64(self.rows // 16)
                        + Int64(source_row >> Int32(4))
                    ) * Int64(self.tile_bytes)
                    local_row = source_row & Int32(15)
                    base = fixed[tile_base + Int64(local_row)].to(Uint32)
                    selector = fixed[
                        tile_base
                        + Int64(16)
                        + Int64(local_row * Int32(self.selector_bytes))
                        + Int64(column >> Int32(3))
                    ].to(Uint32)
                    value = base + ((selector >> Uint32(column & Int32(7))) & Uint32(1))
                    packed_values |= self._clamp(value) << Uint32(8 * lane)
                output_offset = (
                    Int64(expert) * Int64(self.columns) + Int64(column)
                ) * Int64(self.rows) + Int64(row_base + packed_in_task)
                output_u32[output_offset >> Int64(2)] = packed_values
                word += Int32(self.threads)
            cute.arch.sync_threads()
            task_offset = Int64(expert) * Int64(self.tasks + 1) + Int64(task)
            cursor = offsets[task_offset].to(Int64) + Int64(tid)
            end = offsets[task_offset + Int64(1)].to(Int64)
            while cursor < end:
                entry = exceptions[cursor].to(Uint32)
                position = entry & Uint32(0xFFFFFF)
                source_row = Int32(position // Uint32(self.columns))
                column = Int32(position % Uint32(self.columns))
                row = source_row
                if cutlass.const_expr(self.rotation):
                    row = (source_row + Int32(self.rows - self.rotation)) % Int32(
                        self.rows
                    )
                low = row & Int32(7)
                high = (row >> Int32(3)) & Int32(7)
                swapped = (
                    (high & Int32(4))
                    | ((high & Int32(1)) << Int32(1))
                    | ((high >> Int32(1)) & Int32(1))
                )
                packed_row = (row & ~Int32(63)) | (low << Int32(3)) | swapped
                if cutlass.const_expr(self.columns == 1):
                    p = (
                        ((row & Int32(6)) << Int32(2))
                        | (row & Int32(1))
                        | ((row & Int32(24)) >> Int32(2))
                    )
                    packed_row = (
                        (row & ~Int32(31))
                        | (p & ~Int32(3))
                        | ((p & Int32(1)) << Int32(1))
                        | ((p >> Int32(1)) & Int32(1))
                    )
                output_offset = (
                    Int64(expert) * Int64(self.columns) + Int64(column)
                ) * Int64(self.rows) + Int64(packed_row)
                output[output_offset] = Uint8(self._clamp(entry >> Uint32(24)))
                cursor += Int64(self.threads)


@functools.cache
def _compiled_packed_scale(rows, columns, task_rows, rotation, clamp, unique, counts=False, ids64=False):
    key = (
        int(rows),
        int(columns),
        int(task_rows),
        int(rotation),
        bool(clamp),
        bool(unique),
        bool(counts),
        bool(ids64),
    )
    launch = _PackedScaleDecode(*key)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int64 if ids64 else cutlass.Int32, 8 if ids64 else 4,
                 cute.AddressSpace.gmem, assumed_align=8 if ids64 else 4),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.x4t_packed_scales", 3, key),
    )


def decode_x4t_packed_scales(
    batch: X4TScaleBatch,
    expert_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    clamp_e8m0_bf16: bool = True,
    expert_ids_unique: bool = False,
    expert_counts: bool = False,
    program=None,
    stream: cuda.CUstream | None = None,
) -> None:
    """Decode a scale plane in one allocation-free launch.

    Output is uint8 ``[E, columns, rows]`` in native W4A16 scale order. Negative
    or out-of-range expert IDs are inactive. Repeated IDs are deduplicated unless
    the caller asserts uniqueness. Exception partitions and row rotation are
    established by ``make_x4t_scale_batch`` at load time. Task rows must be
    multiples of 64 so packed-row permutations remain within CTA ownership.
    With ``expert_counts=True``, the input is instead one routing count per
    expert; positive entries select that expert without scanning route IDs.
    ``program`` may retain the matching load-time compiled callable across
    cache reclamation. Its geometry and mode must match the call arguments.
    """
    batch.validate()
    task_rows = batch.exception_task_rows
    if not task_rows or task_rows % 64 or batch.task_exception_offsets is None:
        raise ValueError("Packed X4T requires exception tasks aligned to 64 rows")
    if expert_ids.dtype not in (torch.int32, torch.int64) or expert_ids.ndim != 1:
        raise TypeError("Packed X4T expert_ids must be one-dimensional int32/int64")
    if expert_counts and expert_ids.numel() != batch.num_experts:
        raise ValueError("X4T route counts must contain one value per local expert")
    if output.dtype != torch.uint8 or tuple(output.shape) != (
        batch.num_experts,
        batch.columns,
        batch.rows,
    ):
        raise ValueError("Packed X4T output must be uint8 [experts, columns, rows]")
    for tensor in (expert_ids, output):
        if tensor.device != batch.fixed.device or not tensor.is_contiguous():
            raise ValueError(
                "Packed X4T buffers must be contiguous on the batch CUDA device"
            )
    if not expert_ids.numel():
        return
    compiled = program if program is not None else _compiled_packed_scale(
        batch.rows,
        batch.columns,
        task_rows,
        batch.exception_row_rotation,
        clamp_e8m0_bf16,
        expert_ids_unique,
        expert_counts,
        expert_ids.dtype == torch.int64,
    )
    compiled(
        make_ptr(
            cutlass.Uint8,
            batch.fixed.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint32,
            batch.exceptions.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int64,
            batch.task_exception_offsets.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        ),
        make_ptr(
            cutlass.Int64 if expert_ids.dtype == torch.int64 else cutlass.Int32,
            expert_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8 if expert_ids.dtype == torch.int64 else 4,
        ),
        make_ptr(
            cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        batch.num_experts,
        batch.exceptions.numel(),
        expert_ids.numel(),
        current_cuda_stream() if stream is None else stream,
    )
