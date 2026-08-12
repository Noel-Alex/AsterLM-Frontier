# Pretraining architecture freeze — 2026-08-11

This document ends the broad AsterLM laptop architecture search. The remaining pretraining work is
launch readiness for one model, not another open-ended sweep.

## Frozen model

- Model config: `configs/model/aster_k3_latentmoe_270m_a188m.yaml`
- Geometry: 269,677,164 total parameters; approximately 188,272,620 active per token.
- Mixer: 3:1 KDA3/MLA hybrid, six 128-dimensional KDA heads, short convolution and safe gating.
- FFN: Stable LatentMoE with 8 routed experts, top-2 routing, 2 shared experts, SiTU-GLU and
  quantile balancing.
- Residual/norm: standard residual path and RMSNorm. Block AttnRes and mHC are not silently enabled;
  they remain post-launch research because current Aster evidence does not justify risking the base
  run.
- MTP: disabled for the frozen base pretraining run. A draft/speculative head is a later bounded
  addition, not a reason to restart backbone selection.

This is the largest current K3 configuration that has demonstrated complete full-state laptop fit
with the quality-leading optimizer family. The prepared 868M, 1.45B and 1.95B shapes remain remote
or future scale-up tiers; they do not replace the laptop model merely because total parameter count
is larger. A larger model must fit its weights, gradients, optimizer, checkpoint serialization and
native context without offload destroying time-to-quality.

## Frozen laptop execution recipe

- Physical MoE backend: packed CUTLASS grouped experts.
- Training precision: BF16 AMP.
- Optimizer: per-head Muon/AdamW, Muon LR `5e-3`, cosine schedule.
- Recovery/control optimizer: AdamW-WSD at `5e-4`.
- Expert parameter storage is packed after device placement and optimizer/checkpoint restore.
  Safetensors serialization temporarily materializes independent views and restores packing.
- CUDA allocator: expandable segments, recorded in every manifest.
- Energy is recorded only as telemetry and never ranks or blocks a run.

The final source-pinned backend matrix used four alternating repetitions at sequence 2,048,
microbatch 2, accumulation 8, five warm-up and twenty measured updates. Packed CUTLASS reached a
9,294.9 tok/s median versus 9,188.1 tok/s for GPU-resident `torch_grouped` (+1.16%), with both at
5.774 GiB peak allocated memory. `torch_grouped` eliminated host routing syncs but lost the time in
backward execution. This closes the laptop backend choice.

The optimizer screen used one source-pinned 4,194,304-token seed. Per-head Muon `5e-3` led quality
at equal wall time (6.1482) and terminal loss (6.0556). AdamW-WSD `5e-4` is the retained control
(6.4310 equal-wall, 6.3715 terminal). Muon's current optimizer share is optimization debt, but its
quality lead means the implementation is optimized without reopening the optimizer tournament.

## Stop rule

No new backbone, expert layout, residual system, optimizer family, precision family or scale tier
is screened before launch readiness unless one of these hard blockers occurs:

1. non-finite loss or gradients under the frozen recipe;
2. failure of exact checkpoint resume;
3. inability to fit the declared native context on the 12 GB laptop;
4. reproducible corruption or a correctness mismatch against the semantic reference;
5. a retained control beats the frozen recipe at a predeclared long-run checkpoint, using the same
   initialization, data order, tokens and wall-time accounting.

Low utilization is an execution bug, not an automatic architecture rejection. Kernel work may
replace the physical backend while preserving model semantics. Likewise, remote Hopper/Blackwell
recipes may use a different execution engine or FP8 kernel after an exact-GPU qualification, but
their checkpoints must remain semantically portable.

## Remaining launch-readiness gates

These are finite pass/fail gates, not candidate sweeps:

1. one full-recipe native-context fit and exact-resume recovery check;
2. sealed tokenizer and cleaned/decontaminated corpus manifest;
3. W&B and private Hugging Face identities plus checkpoint upload/download verification;
4. local checkpoint pyramid and permanent milestone policy verification;
5. final preflight showing clean source, sufficient disk, no competing CUDA process and all
   contracts resolved;
6. a short canary that advances data cursors, saves, resumes exactly and appears in Studio/W&B.

After those gates pass, start stage 1. Architecture research can continue against saved checkpoints
while the frozen base model trains; it must not silently mutate the live run.
