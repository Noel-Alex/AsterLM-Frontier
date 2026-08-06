# AsterLM Frontier audit report

Generated: 2026-08-06

## Scope

The uploaded code archive and download transcript were audited for:

- model/train/data/reasoning config compatibility;
- resumable Hugging Face materialization;
- partial-download and interpreter-shutdown recovery;
- raw-to-clean corpus flow;
- local training data safety;
- tokenizer/reasoning-token compatibility;
- pretraining, SFT, DPO, and RLVR resume behavior;
- command/documentation consistency;
- regression tests and CLI/config validation.

The uploaded archive did not contain the user's large `data/`, `runs/`, or checkpoint artifacts, so their bytes could not be inspected directly. The transcript and repository state formats were used to diagnose the existing pilot. The updated archive is code/config/docs only and must be merged while preserving the local **root** `data/` directory. The rebuilt package explicitly retains `src/asterlm/data/` and `configs/data/`.

## Current data conclusion

The transcript shows these committed pilot totals:

- FineWeb-Edu: approximately 325,001,299 tokens;
- Cosmopedia-v2: approximately 50,000,407 tokens;
- FineMath-4+: approximately 50,000,640 tokens;
- DCLM: approximately 75,005,799 tokens;
- combined materialized total: approximately 500,008,145 tokens.

The post-completion `PyGILState_Release` failures were interpreter-finalization failures after source data and state had been committed. They did not justify deleting the pilot. MMLU then exhausted at 14,042 records while the config demanded 15,000, so that stage could never succeed as written.

## Critical fixes applied


### Packaging and environment isolation

- Corrected an archive-filter bug that had excluded every directory named `data`, including `src/asterlm/data/` and `configs/data/`.
- The rebuilt archive excludes only root runtime data/checkpoints/caches.
- Project-local Hugging Face caches now preserve the token path created by a normal global `hf auth login`.
- The downloader rejects a virtual environment belonging to another checkout unless `--allow-external-venv` is explicitly supplied.
- Deterministic preflight and source-validation failures no longer incur pointless command-level retry delays.

### Materialization and download recovery

- Complete finite benchmark splits now use `target_records: null` rather than arbitrary impossible caps.
- Record targets can be capped or complete-split, and old exhausted MMLU state migrates without redownloading.
- Corpus/record signatures include every transformation-affecting field while excluding expandable token/record caps.
- Existing pilot checkpoints safely migrate from the older signature when source transformation fields match.
- Successful materializers flush output/state and use a direct process exit to avoid the observed pyarrow/datasets CPython teardown crash.
- JSON and pickle state writes are atomic and fsynced.
- Status output handles complete-split record targets.

### Training-data safety

- Generic local readers accept only supported data/text formats.
- Cursor pickle, SQLite, partial, temp, log, state, manifest, report, audit, summary, stats, and metrics files are excluded.
- Missing `data/...` paths are treated as missing local paths, never accidental Hugging Face repository IDs.
- Multi-worker local readers divide files or records without duplicating examples across workers.
- Reasoning data readers use the same control-file exclusions.

### Cleaning and validation

- Cleaning can route a deterministic fraction of accepted unique documents to a separate validation output.
- The generated frontier data config now uses disjoint local validation shards instead of a remote Cosmopedia training stream.
- Older clean outputs without the new holdout format are detected; `--reset-existing` rebuilds only the clean derivative.

### Tokenizer and reasoning integration

- All reasoning and direct-mode special tokens are required by `AsterTokenizer`.
- Tokenizers trained before the reasoning tokens were added fail clearly and must be retrained.
- Current SmolTalk2 split identifiers are corrected.
- Reasoning preparation ignores materializer control files.
- GSPO, DAPO, GRPO, and Dr.GRPO now have separate output roots to prevent cross-algorithm resume/overwrite collisions.

### Resume behavior

- Pretraining now exposes mutually exclusive `--init-checkpoint` and `--resume` semantics.
- SFT `--resume` is no longer ignored.
- DPO can resume model, optimizer, RNG, step, and token state.
- Trainer resume forces single-process loading and deterministically replays consumed packed microbatches to restore the data position. This is exact but may be slow for a very late checkpoint.

### DPO integration

- Offline reference scoring accepts JSON/JSONL, gzip, Zstandard, or a directory of shards.
- UltraFeedback message-list `chosen` and `rejected` values are normalized to the last assistant response.
- Reference-scored output is written atomically.
- New laptop-safe frontier SFT and DPO configs were added.

### Operational readiness

- Added `scripts/training_preflight.py` for config, tokenizer, local data, first-record, dependency, CUDA/BF16, checkpoint, output, and disk checks.
- Added `docs/TRAINING_RUNBOOK.md` with commands for all supported training and post-training paths.
- Replaced stale commands in `START_HERE.md`.

## Validation performed

- `python -m compileall -q src scripts tests`: passed.
- `pytest -q`: **61 tests passed**.
- All 11 model YAML files parse through `AsterConfig`.
- All 22 train YAML files parse through `TrainConfig`.
- All 11 checked-in data YAML files parse through `DataConfig`; the generated clean-data config is created after cleaning.
- All 4 reasoning YAML files parse through `RLVRConfig`.
- All 8 corpus YAML files parse as YAML.
- The complete `download_data --profile all --validate-first --dry-run` command graph is valid.
- Training runbook references were checked against repository paths; `configs/data/pretrain_frontier_clean.yaml` is intentionally generated by the cleaning step.

## Required live-machine validation

The following cannot be proven from the uploaded code-only archive and CPU audit environment:

- byte-level integrity of the user's existing local shards beyond the transcript;
- successful live Hugging Face validation after repository/split changes;
- FLA KDA CUDA behavior;
- TorchAO/APOLLO/Transformer Engine compatibility with the installed Torch/CUDA build;
- actual RTX 4080 Laptop peak VRAM, throughput, and thermals;
- final model quality, long-context quality, and RL reward-hacking behavior.

Run the shard verifier, training preflight, and short VRAM matrix on the actual laptop before a long run.
