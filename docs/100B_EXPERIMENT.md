# AsterLM 100B-token research campaign

This is the authoritative plan for deliberately overtraining the 890M-logical / 482M-active AsterLM MoE candidate on up to **100B training tokens**.

## Scientific position

The 18.4B corpus remains the first compute-optimal checkpoint. The 50B and 100B tiers are not claimed to be compute-optimal; they are controlled overtraining experiments intended to test:

- whether validation loss and downstream quality continue improving after the classic compute-optimal point;
- whether MoE expert pathways become more structured and transferable after training loss begins to flatten;
- whether math, code, long-context and instruction-following quality improve at different rates;
- whether the additional compute is preferable to training a larger model.

The campaign uses mostly unique, deduplicated data. It does **not** manufacture 100B tokens by blindly repeating a small corpus. Exact repetition and source-level effective epochs must remain visible in the corpus audit.

## Progressive data tiers

Every tier expands the same source directories and cursor checkpoints. No completed pilot/frontier shard is downloaded twice.

| Tier | FineWeb-Edu | DCLM | Cosmopedia | FineMath | Stack-Edu | Total |
|---|---:|---:|---:|---:|---:|---:|
| Frontier | 10.4B | 2.4B | 1.4B | 1.8B | 2.4B | 18.4B |
| Overtrain-50 | 28B | 7B | 3B | 5.5B | 6.5B | 50B |
| Overtrain-100 | 54B | 16B | 6B | 11B | 13B | 100B |

The 100B mix remains 54% educational web, 16% DCLM diversity, 6% synthetic exposition, 11% mathematics and 13% permissive code.

## One-command data acquisition

Full 100B campaign, including benchmarks, post-training and reasoning data:

```bash
source .venv/bin/activate
python scripts/data_campaign.py \
  --tier 100b \
  --network-mode low
```

The command validates every remote source, resumes the existing 500M pilot, downloads only missing data, and verifies every finalized compressed shard.

To download only pretraining data:

```bash
python scripts/data_campaign.py --tier 100b --pretraining-only --network-mode low
```

To clean, deduplicate, decontaminate and create deterministic held-out validation shards after download:

```bash
python scripts/data_campaign.py \
  --tier 100b \
  --network-mode low \
  --clean \
  --prune-hf-cache-before-clean
```

`--prune-hf-cache-before-clean` removes only reconstructable Hugging Face cache files after materialization. It does not remove AsterLM raw shards or cursor state. Use it because raw + cache + cleaned copies of a 100B-token corpus can otherwise exceed a 1TB drive.

For a fast integrity check during a long campaign, add `--verify-last-only`. Before final training, run without it so every shard is decompressed and checksum-verified.

## Disk expectations

Compressed size is not token count. Depending on source composition and compression ratio, 100B materialized tokens can occupy several hundred GiB. The campaign preflight conservatively reserves approximately 620GiB for pretraining data and 660GiB for the complete campaign. Cleaning creates another corpus copy; prune the reconstructable HF cache or use a second drive before cleaning.

## Training schedule

The 100B-token curriculum is:

1. **92B tokens at 8K** using `frontier_100b_stage1_8k.yaml`.
2. **6B tokens at 16K** initialized from stage 1.
3. **2B tokens at 32K** initialized from stage 2.

This puts most compute into efficient base pretraining while still genuinely training long context.

### Stage 1

```bash
python scripts/training_preflight.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_100b_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --check-first-record \
  --json runs/preflight-100b-stage1.json

python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_100b_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

Permanent checkpoints are created at 18.4B, 50B and 92B stage-1 tokens. They are protected from rolling local retention. The learning-rate decay occurs only near the end of the 92B stage, so the 18.4B and 50B checkpoints remain useful continuation points rather than prematurely cooled models.

### Stage 2

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_100b_stage2_16k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-frontier-100b-stage1-8k \
  --hub-repo YOUR_HF_USERNAME/AsterLM-Frontier-100B
```

### Stage 3

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_100b_stage3_32k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-frontier-100b-stage2-16k \
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
python scripts/experiment_status.py runs/aster-frontier-100b-stage1-8k
```

## Checkpoint and Hugging Face policy

Local periodic checkpoints retain model, optimizer, RNG, step and token state. The latest rolling checkpoints are kept locally; permanent token milestones and final checkpoints are never deleted automatically.

When `--hub-repo` is supplied, the trainer creates/uses a **private model repository** and uploads:

- each permanent token milestone;
- final checkpoints;
- optimizer/RNG state by default, enabling disaster recovery on another machine;
- run manifest, JSONL metrics, latest pointer and Hub sync state.

Uploads are synchronous at milestones so a successful milestone means both local serialization and remote backup completed. Hugging Face Xet uploads are resumable and deduplicate already-uploaded chunks. A failed upload is logged but does not kill training unless `hub_fail_on_error: true`.

Retry a failed/manual sync:

```bash
python scripts/sync_run_to_hub.py \
  --run runs/aster-frontier-100b-stage1-8k \
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
- throughput, energy and wall-clock cost per unit quality gain.

A loss plateau alone is not a stopping rule if downstream generalization and pathway structure are still improving. Conversely, a training-loss improvement without held-out or downstream gain is not evidence of useful grokking.
