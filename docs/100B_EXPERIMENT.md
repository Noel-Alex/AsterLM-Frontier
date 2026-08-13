# AsterLM 100B-token research campaign

This is the frozen research plan for the 1.448B-total / 568.2M-active Aster K3
model over **100B training tokens**. The corrected 868M/1.45B/1.95B scale gate
selected the largest sustained no-offload laptop model with favorable matched
quality evidence. The machine-readable sources of truth are
`configs/pretraining/frontier_100b_k3.yaml` and
`configs/experiments/pretraining_selection.yaml`.

## Scientific position

The frozen schedule is a 100B-token controlled overtraining experiment, not a claim that 100B is compute-optimal. Permanent milestones retain the earlier 18.4B and 50B analysis points without splitting the run into incompatible campaigns. The experiment tests:

- whether validation loss and downstream quality continue improving after the classic compute-optimal point;
- whether MoE expert pathways become more structured and transferable after training loss begins to flatten;
- whether math, code, long-context and instruction-following quality improve at different rates;
- whether the additional compute is preferable to training a larger model.

The campaign uses mostly unique, deduplicated data. It does **not** manufacture 100B tokens by blindly repeating a small corpus. Exact repetition and source-level effective epochs must remain visible in the corpus audit.

The four `frontier_100b_*` train configs are mechanically `run_class: final`. Directly invoking the trainer cannot bypass the clean-corpus manifest or promotion gate: it will refuse to allocate the model until the corpus is sealed, every required gate has durable passed evidence, the Git checkout is clean, and full private Hugging Face plus W&B continuity is configured.

## Current raw-data ledger

Every tier expands the same source directories and resumable cursors. No completed pilot/frontier shard is downloaded twice. The launch source of truth is `configs/pretraining/frontier_100b_k3.yaml`.

| Source | Materialized raw tokens | State |
|---|---:|---|
| FineWeb-Edu-Dedup | 54,000,000,001 | complete |
| DCLM baseline | 16,000,001,919 | complete |
| Cosmopedia-v2 | 6,000,000,695 | complete |
| FineMath-4+ | 8,031,559,595 | source exhausted below its historical 11B request |
| Nemotron-CC-Math-4+ | 3,000,000,834 | complete candidate pool |
| Nemotron-CC-Code-v1 | 8B target | Hugging Face access awaiting NVIDIA review |
| Nemotron synthetic code v1 | 5B target | Hugging Face access awaiting NVIDIA review |

The current materialized total is 87,031,563,044 raw tokens; the two pinned code sources bring the expected total to 100,031,563,044. Cleaning, cross-source deduplication and benchmark decontamination reduce usable unique tokens, so the trainer records exact effective epochs and intentional weighted replay after one full clean pass. Stack-Edu is permanently retired: its Software Heritage reconstruction path is not part of the launch plan.

## Remaining data acquisition and seal

After the Hugging Face account is approved for both pinned NVIDIA repositories, resume only the missing code sources:

```bash
python scripts/materialize_corpus.py \
  --config configs/corpus/corpus_nemotron_candidates_16b.yaml \
  --only nemotron_cc_code \
  --only nemotron_synthetic_code
```

The materializer commits remote cursor and compressed-shard state, so retries continue from the last durable boundary. Do not rebuild the already complete 3B Nemotron math source.

After both code sources complete, globally clean, cross-deduplicate, decontaminate and create deterministic held-out validation shards:

```bash
python scripts/prepare_frontier_data.py \
  --raw-corpus data/corpus-frontier-16b \
  --raw-code data/corpus-nemotron-candidates \
  --code-id nemotron_frontier \
  --benchmarks data/decontamination-benchmarks \
  --output data/clean-frontier
```

Then train and seal the final tokenizer. The command atomically publishes `tokenizer.json` and a corpus-bound `tokenizer_manifest.json` with hashes, build parameters and source-level fertility measurements:

```bash
python scripts/train_tokenizer.py \
  --data data/clean-frontier/pretrain_data.yaml \
  --documents 2000000 \
  --fertility-documents 1000 \
  --vocab-size 32768 \
  --output artifacts/tokenizer.json \
  --manifest artifacts/tokenizer_manifest.json
```

For a fast integrity check during a long campaign, add `--verify-last-only`. Before final training, run without it so every shard is decompressed and checksum-verified.

## Disk expectations

Compressed size is not token count. Depending on source composition and compression ratio, 100B materialized tokens can occupy several hundred GiB. The campaign preflight conservatively reserves approximately 620GiB for pretraining data and 660GiB for the complete campaign. Cleaning creates another corpus copy; prune the reconstructable HF cache or use a second drive before cleaning.

## Training schedule

The 100B-token curriculum is:

1. **92B tokens at 4K**, full-parameter, using `frontier_100b_stage1_4k.yaml`.
2. **3B tokens at 8K**, context extension initialized from stage 1.
3. **3B tokens at 16K**, context extension initialized from stage 2.
4. **2B tokens at 32K**, context extension initialized from stage 3.

This puts most compute into the measured sustainable laptop geometry while still
genuinely training the attention, recurrent state, routing, and normalization
subsystems at long context. The current exact-gradient 32K KDA path requires a
qualified high-memory CUDA target; it is not claimed to fit the 12 GiB laptop.

The authoritative unattended launcher runs preflight, exactly resumes an interrupted stage, initializes each longer-context continuation from the prior completed checkpoint, forwards stop signals to a safe checkpoint boundary, and refuses to advance without a complete durable final state:

```bash
python scripts/run_pretraining_campaign.py \
  --campaign configs/pretraining/frontier_100b_k3.yaml \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B \
  --verify-manifest-hashes
```

The same control is available in Aster Studio at `http://localhost:8765`. The launch-readiness ledger remains fail-closed until data, tokenizer, evidence, credentials, private Hub repository and pinned Git state are ready.

Promotion is stage-aware. Stage 1 is never circularly blocked by a retrieval
claim that requires a trained stage-1 checkpoint. Before stage 2, the resulting
checkpoint must pass the long-context retrieval/interference gate. The gate is
revalidated against the promoted checkpoint before later context stages; the
campaign cannot silently flow from 4K to 32K on stale proxy evidence.

### Stage 1

```bash
python scripts/training_preflight.py \
  --model configs/model/aster_k3_latentmoe_1p45b_a568m.yaml \
  --train configs/train/frontier_100b_stage1_4k.yaml \
  --data data/clean-frontier/pretrain_data.yaml \
  --check-first-record \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B \
  --json runs/preflight-100b-stage1.json

python scripts/studio_train.py --mode pretrain \
  --model configs/model/aster_k3_latentmoe_1p45b_a568m.yaml \
  --train configs/train/frontier_100b_stage1_4k.yaml \
  --data data/clean-frontier/pretrain_data.yaml \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

Crash recovery is deliberately much denser than the permanent history. Every
stage writes a complete local recovery checkpoint after **30 wall-clock minutes
or 250 optimizer updates, whichever happens first**. At the campaign's fixed
131,072 tokens/update, the step trigger caps exposure at 32,768,000 tokens; the
timer gives the tighter bound on slower hardware. The newest six recovery
checkpoints remain dense and eight exponentially older bands form a logarithmic
history. The selected model has roughly 2.70 GiB of BF16 weights alone, so the
campaign uses a pessimistic 6 GiB full-state planning estimate until the first
real checkpoint manifest measures the optimizer and metadata payload. Six dense
recovery checkpoints project to 36 GiB. The exact 150 GiB local budget remains
authoritative as checkpoint size evolves; SIGTERM/controlled-stop checkpoints
are written immediately.

Stage 1 creates 13 permanent checkpoints at 0.5B, 1B, 2B, 4B, 8B, 12B,
18.4B, 25B, 35B, 50B, 65B, 80B and 92B tokens. Stages 2, 3 and 4 add four, six
and five context-continuation milestones respectively, for 28 permanent research
checkpoints across the campaign. They are protected from rolling local retention
and uploaded to Hugging Face. The learning-rate decay occurs only near the end of
the 92B stage, so intermediate checkpoints remain useful continuation points rather
than prematurely cooled models.

The private Hugging Face repository has a user-declared **7.5 TB decimal hard
ceiling**. Aster uses **7.0 TB** as the operational refusal threshold so an
in-flight upload, metadata, or later artifact cannot cross the hard ceiling.
Pessimistically projecting 6 GiB over all 28 permanent pretraining milestones is
168 GiB (before Xet deduplication), far below the guard. This estimate is
displayed and must be recalculated from real checkpoint manifests as the run
evolves. Nothing is automatically deleted from Hugging Face merely to save
space; the user will explicitly authorize later remote cleanup. New uploads
must fail closed if their projected total would exceed the operational guard.

### Stage 2

```bash
python scripts/studio_train.py --mode pretrain \
  --model configs/model/aster_k3_latentmoe_1p45b_a568m.yaml \
  --train configs/train/frontier_100b_stage2_8k.yaml \
  --data data/clean-frontier/pretrain_data.yaml \
  --init-checkpoint runs/aster-frontier-100b-stage1-4k \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

### Stage 3

```bash
FLA_DISABLE_BACKEND_DISPATCH=1 python scripts/studio_train.py --mode pretrain \
  --model configs/model/aster_k3_latentmoe_1p45b_a568m_longctx.yaml \
  --train configs/train/frontier_100b_stage3_16k.yaml \
  --data data/clean-frontier/pretrain_data.yaml \
  --init-checkpoint runs/aster-frontier-100b-stage2-8k \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

### Stage 4

```bash
FLA_DISABLE_BACKEND_DISPATCH=1 python scripts/studio_train.py --mode pretrain \
  --model configs/model/aster_k3_latentmoe_1p45b_a568m_longctx.yaml \
  --train configs/train/frontier_100b_stage4_32k.yaml \
  --data data/clean-frontier/pretrain_data.yaml \
  --init-checkpoint runs/aster-frontier-100b-stage3-16k \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

Use `--resume RUN_OR_CHECKPOINT` only for interruption recovery within the same stage. Use `--init-checkpoint` when starting a new context-length stage because optimizer and schedule state intentionally restart.

## Logging and grokking observability

Every run records:

- JSONL metrics suitable for exact offline analysis;
- TensorBoard events;
- Weights & Biases metrics and full resolved configs;
- train/main/MTP/router auxiliary/router z losses;
- unclipped and clipped gradient norms;
- parameter RMS/max values by experts, router, KDA, global attention and embedding/head;
- expert load min/max ratios, coefficient of variation and routing-bias magnitude;
- MoE pathway adjacent-layer consistency, pair similarity, unique-path fraction and normalized layer entropy;
- QK clipping statistics;
- throughput, data wait, forward, backward and optimizer timings;
- host RAM/CPU, process RSS, GPU memory fragmentation, temperature, clocks, power and utilization;
- estimated training TFLOP/s, cumulative FLOPs, progress, ETA and tokens per logical/active parameter;
- run manifest with Git commit, dirty state, package/hardware information and resolved model/data/train configs;
- failure diagnostic ZIPs containing the final metrics window and system state.

The pathway metrics are monitoring signals inspired by practical MoE grokking research; they are not labeled as an exact reproduction of another paper's definitions. Downstream held-out evaluation remains authoritative.

Inspect a run at any time:

```bash
python scripts/experiment_status.py runs/aster-frontier-100b-stage1-4k
```

## Checkpoint and Hugging Face policy

Local periodic checkpoints retain model, optimizer, RNG, scheduler, scaler, data cursor, step and token state. Each
stage keeps the six newest recovery points plus up to eight exponentially widening
historical bands. This is dense near the live training head and increasingly sparse
farther back, so interruption loss stays bounded without retaining every periodic
checkpoint. An unverified permanent milestone is never deleted automatically; a hash-verified Hub milestone may leave the local cache when the 150 GiB ceiling requires it. The final-run contract requires `checkpoint_policy: full`; the
metrics-only policy used by disposable architecture tests cannot be used for a final
run.

When `--hub-repo` is supplied, the trainer creates/uses a **private model repository** and uploads:

- each permanent token milestone;
- final checkpoints;
- optimizer/RNG state by default, enabling disaster recovery on another machine;
- run manifest, analysis schema, experiment identity, JSONL metrics, TensorBoard/diagnostic artifacts, latest pointer and Hub verification state.

Milestone uploads are synchronously verified before they are declared durable.
Remote provider execution additionally applies a five-minute full-state checkpoint
interval and bounded asynchronous Hub queue, so GPU work can continue while each
upload is verified; a graceful stop waits for its durable optimizer-boundary
checkpoint. Hugging Face Xet uploads are resumable and deduplicate already-uploaded
chunks. Final configs require `hub_fail_on_error: true`; an unverified milestone
can never be declared durable or evicted to satisfy the 150 GiB local cache ceiling.

Retry a failed/manual sync:

```bash
python scripts/sync_run_to_hub.py \
  --run runs/aster-frontier-100b-stage1-4k \
  --repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

Do not automatically upload the raw 100B corpus. Most source data already lives on Hugging Face; mirroring it wastes bandwidth/storage and can complicate redistribution obligations. Upload source manifests, revisions, cleaning reports, tokenizer, model checkpoints and experiment artifacts. Publish a cleaned dataset only after provenance/license review, preferably to a separate private dataset repository first.

## Decision gates

At 18.4B and 50B, compare the permanent checkpoint against the previous tier using identical evaluation code. Continue only if the aggregate evidence supports it:

- held-out loss by source/domain;
- MMLU/ARC/HellaSwag/TruthfulQA;
- GSM8K and math exact match;
- HumanEval/MBPP pass@1;
- long-context retrieval and natural-document perplexity;
- expert load/pathway stability;
- memorization and contamination probes;
- throughput and wall-clock cost per unit quality gain; record energy separately as non-decision telemetry.

A loss plateau alone is not a stopping rule if downstream generalization and pathway structure are still improving. Conversely, a training-loss improvement without held-out or downstream gain is not evidence of useful grokking.
