# Lossless scale compression for FP4 experts

MXFP4-CSF and NVFP4-CSF retain FP4 weight nibbles and recover the original
scale bytes before native expert computation. Both represent ordinary scale bytes as a row base plus an
unsigned offset, with exact exception bytes for values outside the interval.

| Representation | Source scale bytes | Offset width | Native expert arithmetic |
| --- | --- | ---: | --- |
| MXFP4-CSF | E8M0 | 1 bit | MXFP4 weights, BF16 activations |
| NVFP4-CSF | E4M3 | 4 bits | NVFP4 weights and calibrated FP4 activations |

One base belongs to a row of block scales. A row contains multiple scales;
the base is not a replacement for the row's entire scale tensor.

Use `b12x.moe.fused_moe.Mxfp4CsfWeights` or `Nvfp4CsfWeights` with the
ordinary `plan_weights` / `prepare_weights` interface. Supply decoded scale
scratch buffers owned by the caller. Serialized layer executions may reuse
these buffers. Concurrent execution streams require independent scratch.

The tensor loaders are
`b12x.moe.checkpoints.mxfp4_csf.load_mxfp4_csf_weights` and
`b12x.moe.checkpoints.nvfp4_csf.load_nvfp4_csf_weights`. Each takes an iterable
of three `b12x.moe.checkpoints.csf.CsfMatrix` sources per expert, explicit
geometry and a TP rank/size. MXFP4 expects gate/up/down order; NVFP4 expects
up/gate/down order. Each source contains a lazy packed-weight view and CPU
fixed/exception scale tensors. NVFP4 additionally requires the original FP32
scalar weight and activation calibration.

These loaders know tensor geometry and compression layout, but have no model
names, paths, manifests or file ownership. The caller keeps the tensor store
open until loading returns. Weight views are sliced before materialization;
only one rank's packed weights and compressed scale batches are uploaded.
The iterable is consumed once and must contain exactly the declared expert
count. The resulting weights use the existing `plan_weights` /
`prepare_weights` API. Tensor-loading alignment does not extend the serving
geometries accepted by the MoE kernel planner.

The vLLM `mxfp4_csf_loader` and `nvfp4_csf_loader` integrations own checkpoint
validation, supported model inventories, tensor-name mapping and shard
lifetimes. Updating to this tensor-source API requires the matching vLLM
integration; checkpoint bytes, manifests and CLI flags are unchanged.

GPU decoding writes the native scale layout directly. Exception ranges are
partitioned at load time; a thread block patches only its output rows.
The paired NVFP4 decoder expands FC1 and FC2 in one launch. Preparation
retains the integer routing ABIs before graph capture, and replay uses
caller-owned allocations.

## Serialized formats

The vLLM checkpoint readers accept `lil-mxfp4-csf-checkpoint/1` and
`lil-nvfp4-csf-checkpoint/1`, respectively. Scale tensor components use
`.mxfp4_csf_fixed` / `.mxfp4_csf_exceptions` or
`.nvfp4_csf_fixed` / `.nvfp4_csf_exceptions` suffixes. Predecessor schemas
and API aliases are not accepted. Migrate their headers, manifests and
receipts before loading; the compressed payload values do not need refitting.

MXFP4 and NVFP4 describe different source arithmetic. A shared compressed
storage concept does not make their weight, scale, or activation formats
interchangeable. The NVFP4 checkpoint reader supports GLM-5.3-Flash and
Qwen3.8-Flash-Next geometry (TP extents must contain a multiple of 64
intermediate channels);
the MXFP4 reader supports Kimi-K3 and DeepSeek-V4.1-Flash geometry.
