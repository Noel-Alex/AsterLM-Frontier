# Bounded Hugging Face resume fix

## Failure that this fixes

With `datasets==5.0.1`, `IterableDataset.shuffle(buffer_size=1)` may still interleave up to ten input shards. Each active Parquet reader can hold decoded Arrow data, so a 32 GiB laptop can fill RAM and continue processing a large offline backlog after Wi-Fi is disabled. The saved cursor also has ten active child streams, making restart expensive.

## New behavior

- New downloads reshard Parquet inputs by row group when supported.
- Only one remote input shard is active at a time (`max_buffer_input_shards=1`).
- Existing ten-stream cursor files are migrated in place and drained one child at a time. Existing local `.jsonl.zst` shards and token totals are not discarded.
- Every progress bar displays process RSS, system available RAM, and stream layout.
- On a 32 GiB host the automatic RSS ceiling is roughly 20 GiB; a system-available-memory floor is also enforced.
- A memory-pressure stop commits the cursor and local shard, exits with code 75, and is not automatically retried by `download_data.py`.
- Parquet/Arrow allocations are explicitly released after a failed iterator is replaced.

The first restart of an old cursor should print:

```text
fineweb_edu: stream_layout=legacy-multistream-sequential-v1
```

Subsequent checkpoints use an AsterLM-owned sequential cursor. Brand-new sources print:

```text
stream_layout=bounded-reshard-max1-v1
```

## Inspect before restarting

```bash
python scripts/inspect_download_cursor.py \
  data/corpus-frontier-16b/fineweb_edu
```

A legacy affected checkpoint should report `legacy-hf-multistream`, usually with 10 streams.

## Safe restart on a 32 GiB laptop

```bash
python scripts/download_data.py \
  --profile overtrain100 \
  --require-auth \
  --network-mode low \
  --max-retries 0 \
  --max-rss-gib 20
```

Do not delete `state.json`, committed `cursor-*.pkl`, finalized `.jsonl.zst` shards, or `data/hf-cache` before this migration.

Some processing may continue briefly after connectivity disappears because the currently decoded Parquet row group is already local. It should now be bounded to one active row-group/shard path rather than ten concurrent readers and should not consume nearly all system RAM.
