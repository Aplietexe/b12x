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

Checkpoint readers are
`b12x.moe.checkpoints.mxfp4_csf.read_mxfp4_csf_layer` and
`b12x.moe.checkpoints.nvfp4_csf.read_nvfp4_csf_layer`. They slice the
TP-independent compressed payload into a rank's exact extent at load time.
The NVFP4 reader also preserves the original FP32 global and activation
calibration. Its FC1 tensors use kernel-native up/gate order.

GPU decoding writes the native scale layout directly. Exception ranges are
partitioned at load time; a thread block patches only its output rows.
The paired NVFP4 decoder expands FC1 and FC2 in one launch. Preparation
retains the integer routing ABIs before graph capture, and replay uses
caller-owned allocations.

## Compatibility

`X4TWeights` is an identity alias for `Mxfp4CsfWeights`. The
`exact_mxfp4.read_exact_mxfp4_layer` reader remains available. MXFP4 serialized
schema and codec identifiers, tensor suffixes, and compressed bytes do not
change. Existing checkpoints require no re-encoding.

MXFP4 and NVFP4 describe different source arithmetic. A shared compressed
storage concept does not make their weight, scale, or activation formats
interchangeable. The NVFP4 checkpoint reader supports GLM-5.3-Flash geometry;
the MXFP4 reader supports Kimi-K3 and DeepSeek-V4.1-Flash geometry.
