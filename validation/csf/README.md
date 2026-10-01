# CSF tensor-source loading qualification

The B12X CSF loaders consume lazy tensor sources and explicit geometry. vLLM
owns checkpoint manifests, model inventories, tensor names and shard lifetimes.
This separates model integration from TP slicing and GPU tensor preparation.
MXFP4 sources use gate/up/down projection order; NVFP4 sources use up/gate/down
order and retain their original FP32 calibration.

**Implemented:** `CsfMatrix`, `load_mxfp4_csf_weights` and
`load_nvfp4_csf_weights` replace the B12X path-based reader interface. The vLLM
CSF model loaders validate the checkpoint and feed these sources in one pass.
The API change requires updating both packages together. Canonical CSF
checkpoint schemas, tensor bytes, serving flags, scratch ownership and compute
kernels are unchanged. Model and execution restrictions remain enforced by the
vLLM integration and B12X preparation; accepting a tensor geometry in the loader
does not qualify another model or parallel mode.

## Conditions and measurement

The control readers are from B12X
`1b32d927bab8d354718bfb559fa925f23e6266d2`; the vLLM integration starts from
`096c7037ee3f526434ee4816fc555b5cd3767715`. Both arms load the same immutable
checkpoint, one complete expert layer and the same TP rank on the same GPU.
Every returned weight, compressed scale stream, exception table, calibration
tensor and scalar metadata field is compared. Tensor comparison uses raw bytes,
including floating-point calibration bits. Expanded-scale scratch is not
initialized by loading; both arms must retain the exact caller-provided objects.

The environment is Frank1 GPU 10, RTX PRO 6000 Blackwell 96 GB,
UUID `GPU-6171baff-cc22-608e-4029-507f67c392ff`, 600 W. Tests use CUTLASS DSL
4.7.1 and native extensions from image
`sha256:18b00b38c792463e8a9ba886d68b9f58256544776856d67c5146d9b2c872cf3f`,
with the review worktrees on `PYTHONPATH`. SM/memory clocks are dynamic;
these are correctness checks with no throughput or latency claim.

## Results and limits

**Qualified:** all ten full-layer/rank comparisons are byte-identical.

| Published checkpoint | HF main revision | Layer | TP | Ranks |
| --- | --- | ---: | ---: | --- |
| DeepSeek-V4.1-Flash-MXFP4-CSF | `872da235166458bd6ffa9ee3f3c5c4771b63159c` | 0 | 4 | 0, 3 |
| GLM-5.3-Flash-NVFP4-CSF | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` | 3 | 2 | 0, 1 |
| Qwen3.8-Flash-Next-NVFP4-CSF | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` | 0 | 2 | 0, 1 |
| Kimi-K3-MXFP4-CSF | `8b7d43b0f7141c4ff04a5f87d26c8675c30ff7a0` | 1 | 16 | 0, 15 |
| Kimi-K3-MXFP4-CSF | `8b7d43b0f7141c4ff04a5f87d26c8675c30ff7a0` | 1 | 12 | 0, 11 |

All checkpoint IDs are in the `local-inference-lab` HF organization. The Kimi
fixture is one 16,209,974,572-byte source shard copied from Frank2. Its SHA256
matches the published manifest:
`02182faabfa61a4ef6c506d3d1f4c9c905b80fa9e759401dcb632e1d0af74c8e`.
The fixture does not constitute a second complete checkpoint.

**Qualified components:** 88 B12X tests pass, covering arbitrary tensor-source
geometry, both TP extents, projection order, exceptions crossing slice boundaries,
FP32 calibration under BF16 defaults, malformed calibration and expert counts,
GPU scale decoding, native MoE output and captured replay. Thirty vLLM tests pass,
including retained tensors, manifest rejection, file-backed embeddings, shard
reuse, and handle closure after preparation failure. All applicable staged-file
vLLM hooks, manual `mypy-3.12`, and changed-file B12X Ruff checks pass.

**Unsupported by this qualification:** full-model generation, full-vocabulary
KLD, loader latency and serving throughput for the refactored package pair.
No running serving container was changed. The byte comparisons demonstrate
unchanged layer inputs to preparation; they do not establish a speedup or extend
the supported serving matrix.

[Qualification identities](tensor-source-evidence/qualification.json) record
the SHA256 of every changed runtime source file. The adjacent per-case JSON
receipts record hashes of every compared tensor. The archived capture script
records the original `/work`, `/reference` and `/hf` mount layout; use the
portable command below outside that layout.

## Reproduction

Use a matching CUDA/B12X/vLLM environment. The full component commands are in
`tensor-source-evidence/qualification.json`. To compare one Qwen rank, set
`--checkpoint` to its immutable snapshot directory and extract the control reader
from the B12X revision listed above:

```bash
git show 1b32d927bab8d354718bfb559fa925f23e6266d2:b12x/moe/checkpoints/nvfp4_csf.py > /tmp/nvfp4-csf-reference.py
.venv/bin/python validation/csf/compare_tensor_loaders.py \
  --reference-reader /tmp/nvfp4-csf-reference.py \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4-CSF \
  --checkpoint-id local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF@4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e \
  --codec nvfp4 --layer 0 --num-experts 512 \
  --hidden-size 2560 --intermediate-size 640 --tp 2 --ranks 0 1 \
  --output /tmp/qwen-csf-layer-parity.json
```

The portable tool was also executed for Qwen TP2 rank 1; all twelve returned
tensors match. The output file must not already exist. No weights are modified.
