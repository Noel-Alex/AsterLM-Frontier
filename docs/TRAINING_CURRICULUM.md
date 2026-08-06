# Training curriculum

## Selection before scale

The repository intentionally separates **fit tests** from **quality tests**.

### Fit tests

`run_frontier_experiments.py` launches each case in a clean process and records:

- success/OOM
- peak allocated and reserved VRAM
- tokens/s and step breakdown
- optimizer and activation-offload choices
- system telemetry

### Quality tests

`run_quality_ablations.py` trains candidates for equal tokens and ranks held-out loss. Close decisions require more tokens and multiple seeds.

The largest model that fits is not automatically the best model. The final architecture must improve validation/downstream quality enough to justify slower training and routing complexity.

## Suggested staged run

### Phase 0 — pilot and instrumentation

- 500M-token pilot corpus
- 670M MoE and 661M dense candidates
- 2K sequence
- 25M–100M tokens per candidate
- verify loss decreases, logging works, checkpoints resume, experts balance

### Phase 1 — general pretraining

Use 4K or 8K blocks, depending on the winning memory configuration.

- broad web-heavy mix with math/code
- WSD schedule
- MTP enabled
- FIM for a portion of code
- sequence packing
- periodic held-out web/math/code evaluation

The supplied 890M recipe targets up to 16B tokens at 8K. The 1.50B recipe begins with a safer 4B-token 4K stage and should only be extended after fit and quality evidence.

### Phase 2 — mixture refinement and 16K

- lower learning rate
- 16K variable-length or packed long documents
- increase math/code only when domain metrics justify it
- retain general data to avoid catastrophic specialization

### Phase 3 — genuine 32K adaptation

- 750M-token example budget
- CPU-offloaded optimizer in the supplied conservative config
- activation offload
- frequent QK clipping and long-context validation
- natural documents, long code and synthetic retrieval tasks

This stage is what earns the “32K native” claim.

### Phase 4 — reasoning continued pretraining

Use verified math/code/science data and lower learning rate. Keep a general-data fraction. Measure base perplexity and broad-task regression.

### Phase 5 — SFT

- response-only loss
- broad assistant behavior first
- math/code/reasoning subsets later
- small multilingual/tool fraction
- inspect formatting and refusal behavior

### Phase 6 — DPO

Precompute reference log-probabilities so a second reference model does not occupy VRAM. Use conservative beta and a chosen-NLL regularizer if needed. Measure base capability regression.

## Optimizer paths

### Muon + AdamW

Quality/stability reference:

- Muon: ordinary dense hidden matrices and OSP embedding projections
- AdamW no decay: embeddings, tied head, norms, biases, routers, recurrence constants
- AdamW decay: remaining non-Muon matrices

### APOLLO/APOLLO-Mini

Low-rank optimizer-state option for fitting larger full-rank models. It is an external optional dependency; test quality against Muon/AdamW before committing.

### TorchAO AdamW4bit/8bit

Compresses optimizer state. This directly targets persistent VRAM. Kernel/version support must be verified on the installed Torch/CUDA stack.

### CPU-offloaded AdamW

Largest VRAM reduction, potentially severe PCIe slowdown. Appropriate when fitting matters more than time, especially during 32K adaptation.

### LoQT-style INT4 base

FFN/expert full matrices are packed INT4 buffers; only low-rank updates are trainable. Backward reconstructs base tiles rather than keeping full BF16 copies. Updates are periodically merged and requantized, optionally on CPU.

This path should be treated as a separate training algorithm, not a transparent memory switch. Compare its loss trajectory against the BF16/FP8 reference.

## Precision

- BF16 trainable storage is the reference.
- FP8 executes eligible Transformer Engine linears while maintaining scaling metadata.
- sensitive norms, routing and recurrent constants remain higher precision.
- no native FP4 training claim is made for Ada.

## Memory order of operations

When a configuration OOMs:

1. micro-batch 1
2. checkpointing on
3. chunked vocabulary loss on
4. APOLLO-Mini or 4/8-bit optimizer state
5. activation offload
6. CPU optimizer offload
7. LoQT INT4 FFN/expert storage
8. reduce sequence length
9. reduce model only after memory methods are exhausted

This ordering follows the user’s stated priority: slower is acceptable; failing to fit is not.

## Schedule

Warmup–Stable–Decay is default:

- warmup for optimizer stabilization
- long stable LR region
- final decay over a configurable fraction of the actual token/step horizon

The scheduler derives its horizon from `max_tokens` and tokens per step when possible. Phase transitions load weights only and start a fresh optimizer/schedule. Exact resume restores optimizer, RNG and counters.

## Logging

Every run emits:

- resolved model/train/data configs
- git/environment manifest
- loss and perplexity
- per-source validation
- MTP losses
- router losses/load
- gradients/update norms
- QK clip statistics
- KDA gate/state statistics where available
- throughput and timing components
- allocated/reserved/peak VRAM
- CPU RAM
- GPU temperature, power and clocks
- checkpoint duration and size
- diagnostic bundle on failures

Use these outputs to make the next configuration decision rather than manually copying terminal snippets.


## Reasoning-model extension

The `reasoning` download profile adds three independently resumable sources under `data/reasoning-frontier`: Mixture-of-Thoughts for cold-start traces, DAPO-Math for exact-answer RL prompts, and the decontaminated tested Python set for executable-code RL. The `all` profile includes these stages without invalidating existing corpus checkpoints. Full preparation and training instructions are in [REASONING_MODEL.md](REASONING_MODEL.md).
