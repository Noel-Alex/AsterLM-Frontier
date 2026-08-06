# Download reliability fix

This revision keeps the existing pilot reusable and fixes the two failures visible in the supplied transcript.

## Apply to an existing checkout

Copy/merge the updated code over the checkout while preserving `data/`, `runs/`, and `artifacts/`:

```bash
cd ~/Documents/AsterLM-Frontier
source .venv/bin/activate
bash scripts/repair_data_environment.sh
hf auth login
python scripts/download_status.py
python scripts/verify_data_shards.py data/corpus-frontier-16b --only-last
```

Then resume the same pilot:

```bash
python scripts/download_data.py \
  --profile pilot \
  --validate-first \
  --require-auth \
  --network-mode low \
  --max-retries 0
```

Do not delete `data/hf-cache`, source `state.json`, `cursor-*.pkl`, or finalized `*.jsonl.zst` shards.

## What changed

- The observed post-success CPython/pyarrow shutdown crash no longer turns a committed source into a failed stage.
- Complete benchmark splits use `target_records: null`; MMLU is no longer forced to reach an impossible 15,000-record cap.
- Source transformation signatures are versioned and safely migrate existing compatible pilot checkpoints.
- DCLM uses the official Parquet mirror.
- `zstandard` remains an explicit dependency and shard verification reads frame checksums.
- Every Hugging Face stream saves its shard/row cursor.
- Local output shards and state updates are atomic.
- Transient network failures use exponential backoff.
- Each source has a separate log and checkpoint.
- Pilot and frontier downloads share one progressive corpus directory.
- Low-bandwidth mode increases timeouts and reduces Xet concurrency.

See `docs/LOW_BANDWIDTH_DOWNLOADS.md` and `docs/TRAINING_RUNBOOK.md`.
