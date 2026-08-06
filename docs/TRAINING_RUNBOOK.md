# AsterLM Frontier: authoritative training runbook

This is the command source of truth for the audited repository. Run commands from the repository root. Paths under the **root** `data/`, `runs/`, and `artifacts/` directories are intentionally not included in the distributable code archive; source-code directories such as `src/asterlm/data/` and `configs/data/` are included.

Never run one checkout while another checkout's virtual environment is active. The downloader now rejects that state by default. For the normal installation, work only in `/home/nol/Documents/AsterLM-Frontier` and activate `/home/nol/Documents/AsterLM-Frontier/.venv`.

## 1. Install and verify the environment

Python 3.11 through 3.13 is supported. Your existing Python 3.11 environment can be retained if its checks pass; rebuilding with Python 3.12 is optional. The full laptop stack needs APOLLO, TorchAO, tracking, and reasoning extras.

```bash
PYTHON_BIN=python3.12 bash scripts/setup_linux.sh \
  --with-apollo \
  --with-torchao \
  --with-tracking \
  --with-reasoning
source .venv/bin/activate
hf auth whoami || hf auth login

# These two paths must belong to this same checkout.
printf 'repo=%s\nvenv=%s\npython=%s\n' "$PWD" "$VIRTUAL_ENV" "$(command -v python)"

pytest
python scripts/smoke_train.py --steps 2
python scripts/system_check.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --sequence 2048
```

The reasoning tokenizer tokens are mandatory. A tokenizer trained before `<|thinking|>`, `<|direct|>`, `<think>`, `</think>`, `<answer>`, and `</answer>` were added must be retrained.

## 2. Recover or continue existing downloads

Do **not** delete the approximately 500M-token pilot merely because the old processes printed a fatal shutdown error. The committed data is reusable. First inspect and verify it:

```bash
python scripts/download_status.py
python scripts/verify_data_shards.py data/corpus-frontier-16b --only-last
```

Then rerun the pilot command. Completed sources are detected from `state.json`; incomplete sources resume from their last committed cursor and shard.

```bash
python scripts/download_data.py \
  --profile pilot \
  --validate-first \
  --require-auth \
  --network-mode low \
  --max-retries 0
```

`--max-retries 0` means unlimited **materializer network retries**, not zero retries. Deterministic preflight and validation failures are not retried. Clean interruption is one `Ctrl-C`; the orchestrator gives the child process time to finalize its current shard and cursor.

The downloader keeps large Hub caches under `data/hf-cache` while preserving the token created by the normal global `hf auth login`. You should not need `hf auth login --force`.

### Download profiles

```bash
# Tiny benchmark sets used for decontamination
python scripts/download_data.py --profile benchmarks --validate-first --require-auth --network-mode low --max-retries 0

# Standard instruction and preference datasets
python scripts/download_data.py --profile posttrain --validate-first --require-auth --network-mode low --max-retries 0

# Reasoning SFT/RLVR sources
python scripts/download_data.py --profile reasoning --validate-first --require-auth --network-mode low --max-retries 0

# Expand the retained pilot checkpoints to the full web/math corpus and add Stack-Edu
python scripts/download_data.py --profile frontier --validate-first --require-auth --network-mode low --max-retries 0

# Everything, in a safe progressive order
python scripts/download_data.py --profile all --validate-first --require-auth --network-mode low --max-retries 0
```

Monitor and verify at any time:

```bash
python scripts/download_status.py
python scripts/verify_data_shards.py --only-last
# Full verification is slower but reads every compressed frame:
python scripts/verify_data_shards.py
```

The benchmark materializer now treats `target_records: null` as “download the complete pinned split.” This is necessary for datasets such as MMLU whose selected split has fewer records than the old arbitrary cap. Before the first corrected benchmark run, `download_status.py` can still display the legacy `14.04K/15.00K` target stored in the old checkpoint. Running the benchmark profile once migrates that state to complete-split mode without deleting the 14,042 records.

## 3. Clean, deduplicate, decontaminate, and create local holdouts

For the current 500M pilot, omit code until Stack-Edu is downloaded:

```bash
python scripts/prepare_frontier_data.py \
  --raw-corpus data/corpus-frontier-16b \
  --benchmarks data/decontamination-benchmarks \
  --output data/clean-frontier \
  --skip-code
```

For the complete frontier corpus:

```bash
python scripts/prepare_frontier_data.py \
  --raw-corpus data/corpus-frontier-16b \
  --raw-code data/stack-edu-frontier-2p4b \
  --benchmarks data/decontamination-benchmarks \
  --output data/clean-frontier
```

The cleaner writes deterministic, disjoint local validation holdouts under `data/clean-frontier/validation/` and generates:

```text
configs/data/pretrain_frontier_clean.yaml
```

If `data/clean-frontier` was created by an older repository version that did not make local holdouts, rebuild it deliberately:

```bash
python scripts/prepare_frontier_data.py \
  --raw-corpus data/corpus-frontier-16b \
  --raw-code data/stack-edu-frontier-2p4b \
  --benchmarks data/decontamination-benchmarks \
  --output data/clean-frontier \
  --reset-existing
```

This deletes only the clean derivative, not the downloaded raw corpus.

## 4. Train the tokenizer

Train once after the clean mixture exists. This version includes all ordinary, FIM, tool, and reasoning mode tokens.

```bash
python scripts/train_tokenizer.py \
  --data configs/data/pretrain_frontier_clean.yaml \
  --output artifacts/tokenizer.json \
  --vocab-size 32768 \
  --documents 1000000
```

## 5. Run the mandatory preflight and VRAM probes

```bash
python scripts/training_preflight.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --check-first-record \
  --json runs/preflight-frontier-stage1.json

python scripts/hardware_probe.py --output runs/hardware-probe.json
python scripts/run_frontier_experiments.py --mode quick --steps 3
```

For the full architecture/optimizer matrix:

```bash
python scripts/run_frontier_experiments.py --mode full --steps 3
```

For one explicit profile:

```bash
python scripts/profile_training.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train-config configs/train/frontier_stage1_8k.yaml \
  --sequence 8192 \
  --batch 1 \
  --accum 16 \
  --steps 3 \
  --warmup 1 \
  --optimizer apollo_mini \
  --activation-offload \
  --json runs/profile-frontier-8k.json
```

## 6. Pretraining commands

### 6.1 Small 220M development model

```bash
python scripts/training_preflight.py \
  --model configs/model/aster_220m.yaml \
  --train configs/train/pretrain_laptop.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --check-first-record

python scripts/train_pretrain.py \
  --model configs/model/aster_220m.yaml \
  --train configs/train/pretrain_laptop.yaml \
  --data configs/data/pretrain_frontier_clean.yaml
```

Resume the same run, including optimizer, RNG, counters, and deterministic packed-data replay:

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_220m.yaml \
  --train configs/train/pretrain_laptop.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --resume runs/aster-220m-pretrain
```

Continue it at 8K with a fresh optimizer/schedule but loaded weights:

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_220m.yaml \
  --train configs/train/pretrain_8k_laptop.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-220m-pretrain
```

### 6.2 Main 893M logical / 484M-active frontier candidate

Stage transitions use `--init-checkpoint` because they intentionally start a new optimizer and schedule. Interrupted stages use `--resume`.

```bash
# Stage 1: 8K
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml

# Exact same-stage resume
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --resume runs/aster-frontier-stage1-8k

# Stage 2: 16K
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage2_16k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-frontier-stage1-8k

# Stage 3: genuinely trained 32K
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage3_32k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-frontier-stage2-16k
```

### 6.3 LoQT/INT4-FFN experiment

Do not mix this checkpoint with the ordinary BF16 model config.

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_frontier_893m_loqt.yaml \
  --train configs/train/frontier_stage1_8k_loqt.yaml \
  --data configs/data/pretrain_frontier_clean.yaml
```

### 6.4 Larger 1.5B target experiment

Only proceed after a real VRAM profile passes.

```bash
python scripts/train_pretrain.py \
  --model configs/model/aster_moe_target_1p51b_a623m.yaml \
  --train configs/train/frontier_target_stage1_4k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml

python scripts/train_pretrain.py \
  --model configs/model/aster_moe_target_1p51b_a623m.yaml \
  --train configs/train/frontier_target_stage2_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --init-checkpoint runs/aster-target-stage1-4k
```

### 6.5 Equal-token architecture ablations

```bash
python scripts/run_quality_ablations.py \
  --data configs/data/pretrain_frontier_clean.yaml \
  --tokens 100000000 \
  --continue-on-error
```

## 7. Ordinary instruction post-training

Download the post-training profile first. The audited local SFT mixture is `configs/data/sft_frontier_local.yaml`.

```bash
python scripts/training_preflight.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/sft_frontier.yaml \
  --data configs/data/sft_frontier_local.yaml \
  --checkpoint runs/aster-frontier-stage3-32k \
  --check-first-record

python scripts/train_sft.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/sft_frontier.yaml \
  --data configs/data/sft_frontier_local.yaml \
  --checkpoint runs/aster-frontier-stage3-32k
```

Resume SFT:

```bash
python scripts/train_sft.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/sft_frontier.yaml \
  --data configs/data/sft_frontier_local.yaml \
  --resume runs/aster-frontier-sft
```

## 8. DPO with an offline reference

The materializer writes UltraFeedback as a directory of compressed shards. The reference scorer now accepts that directory directly and extracts the last assistant response from message-list `chosen`/`rejected` values.

```bash
python scripts/precompute_dpo_reference.py \
  --checkpoint runs/aster-frontier-sft \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --tokenizer artifacts/tokenizer.json \
  --input data/posttrain-frontier/ultrafeedback_binarized \
  --output data/posttrain-frontier/ultrafeedback_reference_scored.jsonl \
  --max-length 2048

python scripts/train_dpo.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/dpo_frontier.yaml \
  --checkpoint runs/aster-frontier-sft \
  --data data/posttrain-frontier/ultrafeedback_reference_scored.jsonl \
  --max-length 2048
```

Resume DPO:

```bash
python scripts/train_dpo.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/dpo_frontier.yaml \
  --resume runs/aster-frontier-dpo \
  --data data/posttrain-frontier/ultrafeedback_reference_scored.jsonl \
  --max-length 2048
```

## 9. Reasoning post-training

This requires both `posttrain` and `reasoning` download profiles. The end-to-end runner creates cold-start reasoning SFT data, broad direct-mode SFT data, trains the mode-fusion checkpoint, and then runs crash-resumable verifier RL.

Validate two GSPO cycles first:

```bash
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --reasoning configs/reasoning/rlvr_laptop_gspo.yaml \
  --rl-stop-after 2
```

Continue GSPO from its atomic pipeline state:

```bash
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --reasoning configs/reasoning/rlvr_laptop_gspo.yaml \
  --skip-prepare
```

Run matched alternatives in their own output directories:

```bash
# DAPO
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --reasoning configs/reasoning/rlvr_laptop_dapo.yaml \
  --skip-prepare \
  --rl-stop-after 2

# GRPO
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --reasoning configs/reasoning/rlvr_laptop_grpo.yaml \
  --skip-prepare \
  --rl-stop-after 2

# Dr.GRPO
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --reasoning configs/reasoning/rlvr_laptop_dr_grpo.yaml \
  --skip-prepare \
  --rl-stop-after 2
```

Optional low-learning-rate think/direct fusion refresh after GSPO:

```bash
python scripts/train_sft.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/reasoning_fusion_refresh.yaml \
  --data configs/data/reasoning_mode_fusion_local.yaml \
  --checkpoint runs/aster-reasoning-rlvr-gspo
```

## 10. Evaluation and inference

```bash
python scripts/evaluate_perplexity.py \
  --checkpoint runs/aster-frontier-stage3-32k \
  --data configs/data/pretrain_frontier_clean.yaml \
  --sequence 8192 \
  --batches 32

python scripts/evaluate_reasoning.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --checkpoint runs/aster-reasoning-rlvr-gspo \
  --data data/reasoning/rlvr_prompts.jsonl \
  --samples 4 \
  --limit 100

python scripts/infer.py \
  --checkpoint runs/aster-frontier-dpo \
  --prompt "Explain why the sky is blue." \
  --max-new-tokens 256 \
  --cache-dtype hadamard_int4 \
  --mtp-greedy

python scripts/chat.py --checkpoint runs/aster-frontier-dpo

python scripts/reasoning_chat.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --checkpoint runs/aster-reasoning-rlvr-gspo \
  --mode think \
  --thinking-budget 2048
```

## 11. Export and benchmark

OSP folding is for models with `embedding_projection: true`:

```bash
python scripts/export_osp_merged.py \
  --checkpoint runs/aster-frontier-dpo \
  --output artifacts/aster-frontier-dpo-osp-folded

python scripts/export_torchao.py \
  --checkpoint artifacts/aster-frontier-dpo-osp-folded \
  --output artifacts/aster-frontier-dpo-int4 \
  --mode int4

python scripts/benchmark.py \
  --checkpoint artifacts/aster-frontier-dpo-osp-folded \
  --prompt-tokens 8192 \
  --new-tokens 256 \
  --cache-dtype hadamard_int4

python scripts/benchmark_cache_quantization.py --tokens 32768
python scripts/benchmark_speculative.py \
  --checkpoint artifacts/aster-frontier-dpo-osp-folded \
  --new-tokens 128
python scripts/needle_test.py \
  --checkpoint artifacts/aster-frontier-dpo-osp-folded \
  --lengths 8192,16384,32768
```

## 12. Rules that prevent accidental corruption

- Use `--resume` only for the same model, train config, stage, and output lineage.
- Use `--init-checkpoint`/`--checkpoint` to start a new stage with fresh optimizer state.
- Never train directly on `data/corpus-frontier-16b`; train on the generated clean config.
- Do not point a generic local data source at a materializer root unless it contains supported data shards. Control JSON, cursor pickle, SQLite, partial, and report files are excluded by the reader.
- Do not delete the Hugging Face cache while a materializer is running.
- Keep model YAML and the KDA backend pinned by the checkpoint together.
- Run `training_preflight.py` before every new stage and after changing dependencies, tokenizer, model, train, or data configs.
