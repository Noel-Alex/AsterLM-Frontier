# Project status

Generated: 2026-08-06

## Implemented

### Architecture

- Dense and fine-grained MoE decoder variants from 661M to 1.90B logical parameters.
- 3:1 Kimi Delta Attention / MLA-inspired latent-attention layer pattern.
- Official FLA KDA backend plus pure-PyTorch correctness fallback.
- q-LoRA, compressed latent K/V, separate RoPE channel and absorbed one-token decoding.
- Online chunked-softmax decode over quantized cache chunks.
- DeepSeek-style top-k sigmoid routing, shared experts and bias-based balancing.
- MTP depth 2, QK-Clip, SwiGLU, tied embeddings and optional Block AttnRes.
- Outlier-Safe Single-Scale RMSNorm and orthogonal embedding projections.
- Exact OSP projection folding into inference embedding/head weights.

### VRAM and precision

- BF16 parameter storage on CUDA.
- optional Transformer Engine FP8 modules and autocast.
- Muon+AdamW, APOLLO/APOLLO-Mini, TorchAO AdamW4bit/8bit and CPU-offloaded AdamW.
- gradient checkpointing and saved-tensor CPU activation offload.
- LoQT-style packed INT4 FFN/expert base with low-rank trainable updates and periodic merge.
- BF16/float8/int8/int4/Hadamard-int4 latent cache choices.
- TorchAO inference weight-only INT4/INT8 path.

### Data

- pinned-revision source validation and transactional, shard-cursor-resumable materialization.
- 16B web/math/synthetic corpus plan plus 2.4B permissive Stack-Edu code plan.
- SmolTalk/SmolTalk2/OpenThoughts/UltraFeedback post-training materialization.
- resumable Mixture-of-Thoughts, DAPO-Math and decontaminated tested Python reasoning datasets.
- exact/near deduplication, benchmark decontamination, quality filtering, PII and secret policies.
- deterministic disjoint local train/validation holdouts generated during cleaning.
- source-level reports, provenance, checksummed Zstandard shards, low-bandwidth retries and download telemetry.
- official DCLM Parquet mirror, selected-column streaming, progressive pilot/frontier checkpoints and source integrity verification.

### Training, evaluation and serving

- pretraining, staged long-context adaptation, response-only SFT, teacher distillation and offline-reference DPO.
- unified think/direct reasoning tokens, budgeted reasoning inference and mode-fusion SFT.
- exact on-policy grouped rollouts, verifier scoring and process-isolated frozen-reference scoring.
- GSPO, DAPO, GRPO, Dr.GRPO and REINFORCE-baseline policy updates.
- deterministic math/multiple-choice rewards and Bubblewrap-isolated Python test execution.
- atomic RL cycle commits, exact optimizer resume and duplicate-update recovery.
- WSD/cosine schedules, exact resume with deterministic packed-data replay, and weights-only phase transitions.
- JSONL, TensorBoard and optional W&B logging.
- GPU power/clock/temperature/VRAM telemetry and diagnostic bundles.
- hardware probe, isolated VRAM matrix and equal-token quality-ablation runner.
- perplexity, needle retrieval, cache quantization, throughput and speculative benchmarks.
- exact greedy MTP verifier and external DeepSpec cache/export contract.

## Validation completed in the artifact environment

- Python compilation: passed.
- Pytest: **58 tests passed**.
- forward/backward/MTP/optimizer smoke paths: passed.
- cached versus full latent-attention decoding: passed.
- bounded and quantized cache tests: passed.
- LoQT INT4 packed-base backward and merge tests: passed.
- OSP SSNorm and embedding-projection fold equivalence: passed.
- checkpoint round-trip/backend pinning: passed.
- all repository model/train/data/reasoning YAML files parse successfully.
- 890M BF16, 890M LoQT/INT4 and 1.50B meta-device footprint checks: passed.
- BF16/FP8/INT8/INT4/Hadamard-INT4 cache benchmark smoke run: passed.
- complete low-bandwidth download/source-validation command graph dry run: passed.
- cursor checkpoint, legacy-resume, orphan-cleanup and progressive-corpus regression tests: passed.

## Must be validated on the RTX 4080 Laptop GPU

- FLA KDA installation, compilation and numerical behavior.
- Transformer Engine FP8 speed and memory at these matrix sizes.
- TorchAO 4/8-bit optimizer compatibility with the installed Torch/CUDA build.
- APOLLO package behavior and state footprint.
- activation-offload PCIe overhead.
- complete peak VRAM for each model/optimizer/sequence combination.
- thermal throttling and sustained tokens/s.
- quality of the MoE, LoQT and cache-quantized variants after equal-token training.
- 32K, 64K and 128K retrieval/long-code quality.
- remaining live Hugging Face source validation and full corpus materialization beyond the user-reported source checks; the artifact environment did not download the remote datasets.
- reasoning cold-start and RLVR quality curves, reward hacking checks, code-sandbox throughput, think/direct retention and algorithm ablations.

## Reproducibility boundary

Training checkpoints include model, optimizer, random generators, steps and tokens. Data materializers separately pin source revisions and persist Hugging Face iterable-dataset shard/row cursors with atomic local-shard commits. The final training corpus remains cleaned, versioned local shards so training never depends on a changing remote stream.

## Backend compatibility

The FLA and PyTorch KDA implementations have different internal parameters. Checkpoints save the resolved backend and reject incompatible explicit loads. They are not interchangeable without a deliberate conversion procedure.
