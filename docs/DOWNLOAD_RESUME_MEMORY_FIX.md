# Bounded Hugging Face resume fix

## Failure that this fixes

With `datasets==5.0.1`, `IterableDataset.shuffle(buffer_size=1)` can still
interleave up to ten input shards. Hugging Face starts a prefetch future for
each child stream. Each active Parquet reader can retain decoded Arrow data, so
a 32 GiB laptop can fill RAM and continue processing a large offline backlog
after Wi-Fi is disabled.

## Exact checkpoint shape

The affected FineWeb checkpoint contains an outer
`RebatchedArrowExamplesIterable` around a ten-child
`CyclingMultiSourcesExamplesIterable`.

The real Hugging Face 5.0.1 cycling state serializes:

```text
ex_iterable_idx
previous_states
is_exhausted
type
```

It does **not** serialize an `ex_iterables` key.

The outer rebatched wrapper also stores:

```text
previous_state
num_chunks_since_previous_state
cropped_chunk_length
```

On resume, Hugging Face restores `previous_state` and then skips the recorded
number of chunks. Therefore the nested current `examples_iterable` snapshot is
not authoritative. Using it can replay millions of rows.

For checkpoint 208, the exact authoritative values are:

```text
streams:                       10
outer pending chunks:           2
authoritative state:            $.examples_iterable.previous_state
committed documents seen:       4,139,989
committed estimated tokens:     4,847,909,935
```

## New behavior

- New downloads reshard Parquet inputs by row group when supported.
- Only one remote input shard is active at a time
  (`max_buffer_input_shards=1`).
- Existing ten-stream cursor files are migrated without deleting local
  `.jsonl.zst` shards or changing the committed token total.
- Migration restores each child from the cycling state's `previous_states`.
- The outer wrapper's pending chunks are consumed in the original round-robin
  order before switching to one-child-at-a-time sequential draining.
- Child rebatched cursors are iterated through `iter_arrow()` so their own
  `previous_state` and `num_chunks_since_previous_state` values are honored.
- Every progress bar displays process RSS, system available RAM, and stream
  layout.
- On a 32 GiB host, a 20 GiB process RSS ceiling is recommended; a
  system-available-memory floor is also enforced.
- A memory-pressure stop commits the cursor and local shard, exits with code 75,
  and is not automatically relaunched by `download_data.py`.

The first repaired restart of checkpoint 208 should print:

```text
fineweb_edu: stream_layout=legacy-multistream-sequential-v1
```

After the first successful checkpoint, the cursor becomes an AsterLM-owned
sequential cursor. Brand-new sources print:

```text
stream_layout=bounded-reshard-max1-v1
```

## Inspect before restarting

```bash
python scripts/inspect_download_cursor.py   data/corpus-frontier-16b/fineweb_edu
```

Checkpoint 208 should report:

```text
cursor_kind:      legacy-hf-multistream
cursor_streams:   10
resume_skip_chunks:2
resume_state_from:$.examples_iterable.previous_state
```

## Safe restart

Use the included wrapper:

```bash
./RUN_100B_SAFE.sh
```

Its defaults are equivalent to:

```bash
python scripts/download_data.py   --profile overtrain100   --require-auth   --network-mode low   --max-retries 0   --command-retries 0   --max-rss-gib 20
```

The wrapper also stops gracefully before free disk falls below 150 GiB.

Do not delete or edit:

```text
data/corpus-frontier-16b/fineweb_edu/state.json
data/corpus-frontier-16b/fineweb_edu/cursor-00000208.pkl
data/corpus-frontier-16b/fineweb_edu/fineweb_edu-*.jsonl.zst
```

The stale `legacy-single-max1-v1` label in `state.json` is harmless. The cursor
pickle is authoritative, and the repaired detector overrides the stale label.

Some progress may continue briefly after connectivity disappears because one
currently decoded Parquet row group may already be local. It should no longer
open ten remote readers concurrently or accumulate the prior enormous backlog.
