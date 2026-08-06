# AsterLM Frontier

**A complete VRAM-first language-model research and training stack for a 12 GiB RTX 4080 Laptop GPU.**

AsterLM Frontier is not a single untested “best model” configuration. It is a complete laboratory for finding the best model that can actually be trained on the target laptop. It includes large dense and MoE candidates, Kimi Delta Attention, MLA-style compressed global attention, long-context adaptation, low-bit optimizer and stored-weight experiments, quantized latent cache, Outlier-Safe Pre-Training, MTP/speculative decoding, data acquisition and cleaning, post-training, telemetry, evaluations and exact checkpointing.

Start with [START_HERE.md](START_HERE.md), then use the authoritative [training runbook](docs/TRAINING_RUNBOOK.md).

## Core design hypothesis

The current leading candidate is a **3:1 KDA/latent-attention hybrid MoE**:

- 75% Kimi Delta Attention layers with fixed-size recurrent state
- 25% exact content-addressable MLA-inspired global-attention layers
- compressed latent K/V plus a separate small RoPE channel
- eight or twelve fine-grained routed experts, top-2, plus one shared expert
- DeepSeek-style sigmoid routing and bias-based load balancing
- q-LoRA in global attention
- Multi-Token Prediction depth 2
- Muon/QK-Clip-compatible training
- Single-Scale RMSNorm and foldable orthogonal embedding projections
- BF16 reference training, FP8 execution experiment, low-bit optimizer states, activation offload
- optional LoQT-style packed INT4 FFN/expert bases with low-rank updates
- hot BF16 + cold Hadamard-rotated INT4 latent cache

This architecture must beat a 661M dense control before it is accepted.

## Model candidates

Counts below are generated from the current reference parameterization. The official FLA KDA backend has a different internal parameterization; train, resume and serve with the same pinned backend.

| Config | Logical params | Active/token | Approx. model persistence before optimizer | Role |
|---|---:|---:|---:|---|
| `aster_moe_probe_674m_a367m` | 670M | 364M | 1.25 GiB BF16 | Fast MoE/VRAM probe |
| `aster_dense_challenger_666m` | 661M | 661M | 1.23 GiB BF16 | Dense quality control |
| `aster_moe_frontier_893m_a484m` | 890M | 482M | 1.66 GiB BF16 | Main reference candidate |
| `aster_moe_frontier_893m_fp8` | 890M | 482M | backend-dependent | Shape-identical Transformer Engine experiment |
| `aster_moe_frontier_893m_loqt` | 890M | 482M | about 0.93 GiB | Packed INT4 FFN/expert-base experiment |
| `aster_moe_target_1p51b_a623m` | 1.50B | 620M | 2.80 GiB BF16 | Larger VRAM-first target |
| `aster_moe_stretch_1p90b_a769m` | 1.90B | 764M | 3.54 GiB BF16 | Do not use before full VRAM sweep |

These figures exclude optimizer states, gradients and activations. Those are often the real training-memory bottleneck.

## Why MoE is included now

All expert weights still consume storage, so MoE does not magically solve VRAM. It can, however, increase total representational capacity while keeping active compute lower. AsterLM uses a deliberately modest number of experts so each expert receives enough tokens and routing does not dominate a single-GPU workload.

The code logs expert load, routing entropy, load coefficient of variation, routing-bias magnitude, auxiliary and z-losses. MoE is rejected if it loses to the dense control after equal-token training.

## Context and cache

- **Training curriculum:** 4K/8K → 16K → genuinely trained 32K
- **Validated extension target:** 64K–128K
- **Configuration ceiling:** up to 256K for experiments
- **No claim:** a high `max_seq_len` value alone does not mean the model can reason at that length

Most layers use fixed recurrent KDA state. Only global layers retain token-addressable state. Their cache stores one compact latent plus a small RoPE key per token. Old latent chunks can be stored as INT4 after a normalized Hadamard transform; recent tokens remain BF16. The reference decoder performs online softmax over cache chunks rather than reconstructing a full conventional K/V cache.

The cache implementation is **TurboQuant-inspired**, not a claim to reproduce Google’s proprietary/fused kernels or all paper results. The portable implementation prioritizes correctness and memory measurement; a fused Triton/CUDA kernel is the next speed optimization after the cache policy wins quality tests.

## Precision and VRAM strategy

“Four-bit training” is separated into distinct mechanisms:

1. **4-bit optimizer state** through TorchAO AdamW4bit.
2. **4-bit stored FFN/expert base weights** through the LoQT-style experimental backend.
3. **4-bit inference weights** through TorchAO weight-only export/runtime paths.
4. **4-bit latent KV cache** through groupwise or Hadamard-rotated quantization.
5. **FP8 matrix execution** through optional NVIDIA Transformer Engine modules.

The quality reference stores trainable parameters in BF16 on CUDA. The Ada RTX 4080 can test FP8 through Transformer Engine, but native NVFP4 training is a Blackwell feature. Therefore the repository does not pretend that an emulated four-bit GEMM is the same as hardware-native FP4 training.

## Outlier-Safe Pre-Training

Frontier presets support:

- `norm_type: ssnorm`: one learned RMSNorm scale rather than one per channel
- `embedding_projection: true`: two orthogonally initialized, Muon-optimized embedding-space rotations
- `scripts/export_osp_merged.py`: folds the rotations into input/output weights for zero projection overhead at inference

Conventional RMSNorm remains available for controlled comparisons.

## Training methods

- official `fla-core` KDA kernels, with a slow PyTorch correctness fallback
- BF16 parameter storage and autocast
- optional Transformer Engine FP8 linears/autocast
- gradient checkpointing
- saved-tensor CPU activation offload
- chunked/checkpointed vocabulary projection and loss
- exact packed-document overlap and cross-document loss masking
- Muon + AdamW, APOLLO/APOLLO-Mini, TorchAO 4/8-bit AdamW and CPU-offloaded AdamW
- Warmup–Stable–Decay or cosine scheduling based on actual token budget
- QK-Clip and extensive numerical diagnostics
- MTP auxiliary loss
- exact resume of model, optimizer, RNG, steps and tokens, with deterministic packed-data replay
- periodic LoQT merge/requantization and optimizer-state reset

## Data and curriculum

The supplied full pretraining plan targets roughly 18.4B raw tokens:

| Source | Target | Purpose |
|---|---:|---|
| FineWeb-Edu-Dedup | 10.4B | broad high-quality educational web text |
| DCLM baseline | 2.4B | distribution diversity and web coverage |
| Cosmopedia-v2 | 1.4B | synthetic textbook/expository material |
| FineMath-4+ | 1.8B | mathematical language and problem solving |
| Stack-Edu permissive code | 2.4B | Python/C++/JS/TS/Java/SQL/Rust/Go/Shell/C# |

The repository also downloads SmolTalk, selected SmolTalk2 splits, OpenThoughts and UltraFeedback for post-training. Data is validated before materialization, then normalized, filtered, secret/PII checked, deduplicated, benchmark-decontaminated, split into disjoint local train/validation holdouts, provenance-preserved and audited.

Long-running materialization uses pinned source revisions, Hugging Face iterable-dataset shard/row cursors, atomic `state.json` commits, checksummed Zstandard output frames, retry/backoff policies and source-specific logs. DCLM is read from its Parquet mirror, and selected-column streaming avoids downloading fields the materializer discards. The pilot and full frontier plans share checkpoints rather than duplicating downloads.

See [docs/DATA.md](docs/DATA.md), [docs/LOW_BANDWIDTH_DOWNLOADS.md](docs/LOW_BANDWIDTH_DOWNLOADS.md) and [docs/TRAINING_CURRICULUM.md](docs/TRAINING_CURRICULUM.md).

## Thinking model and verifier-backed reinforcement learning

AsterLM now includes a complete reasoning-model post-training stack rather than relying on a prompt-only “think step by step” behavior:

- unified `<|thinking|>` and `<|direct|>` modes in one checkpoint
- explicit `<think>` and `<answer>` delimiters
- a hard inference thinking budget
- verified long-form math/code/science cold-start SFT
- exact grouped on-policy rollout generation with stored old-policy token log-probabilities
- deterministic mathematical equivalence and sandboxed Python test rewards
- GSPO as the MoE-first default, with DAPO, GRPO, Dr.GRPO and REINFORCE-baseline alternatives
- frozen-reference KL scoring in a separate GPU process
- router auxiliary/z regularization during RL and bias-based expert balancing
- dynamic removal of zero-variance groups, repetition/overlong shaping and answer-format checks
- atomic cycle markers preventing a rollout batch from being updated twice after a crash
- optional low-LR think/direct fusion refresh after RL

The rollout, verifier, reference and updater processes run sequentially, so only one full model occupies the 12 GiB GPU at a time. See [docs/REASONING_MODEL.md](docs/REASONING_MODEL.md) and [docs/RLVR_IMPLEMENTATION.md](docs/RLVR_IMPLEMENTATION.md).

Quick path after the pretraining checkpoint exists:

```bash
python scripts/download_data.py \
  --profile reasoning \
  --validate-first \
  --network-mode low \
  --max-retries 0

python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --rl-stop-after 2
```

## Speculative decoding

Implemented:

- two trained MTP future-token heads
- exact greedy self-speculation reference
- acceptance/equality benchmark
- speculative-cache export contract
- DeepSpec integration notes

The Python verifier is a correctness reference, not a fabricated speed claim. Production speed requires cache-aware fused verification, tree/block verification or integration into an engine such as SGLang/vLLM.

## Repository map

```text
configs/
  corpus/       download/materialization plans
  data/         training mixtures
  model/        dense, MoE, FP8 and LoQT candidates
  train/        probe and staged-training recipes
src/asterlm/
  data/         mixture, packing, tokenizer, quality/decontamination
  generation/   cached decode, sampling, MTP speculation
  layers/       KDA, latent attention, MoE, FFN, norms, MTP
  optim/        Muon, hybrid partitioning, schedules
  quantization/ INT4 cache and LoQT-style stored weights
  training/     engine, precision, checkpointing, telemetry
  reasoning/    modes, rollouts, verifiers, GSPO/DAPO/GRPO losses
scripts/        setup, download, clean, train, evaluate, profile, export
integrations/   external serving/drafter contracts
tests/          correctness and regression tests
```

## Quick verification

```bash
PYTHON_BIN=python3.12 bash scripts/setup_linux.sh \
  --with-apollo --with-torchao --with-tracking --with-reasoning
source .venv/bin/activate
python scripts/system_check.py --model configs/model/aster_moe_frontier_893m_a484m.yaml
pytest
python scripts/smoke_train.py
python scripts/data_preflight.py --minimum-free-gib 90
python scripts/download_data.py --profile all --validate-first --network-mode low --dry-run
```

## Honest limitations

- The artifact was validated in a CPU-only environment. FLA CUDA kernels, Transformer Engine FP8, TorchAO CUDA optimizer behavior, thermals, real peak VRAM and throughput must be tested on the RTX 4080 laptop.
- The Hadamard INT4 cache is portable reference code, not yet a fused TurboQuant kernel.
- The LoQT branch is a practical packed-base/low-rank implementation; it is not claimed to reproduce every gradient-SVD refresh detail of the research paper.
- A 1.5B or 1.9B configuration existing in YAML does not prove that its complete optimizer/activation footprint fits.
- No trained checkpoint is included. Final quality depends overwhelmingly on tokens, data filtering, curriculum and ablations.
