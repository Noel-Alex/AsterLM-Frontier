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
- The default `next-shard` migration is performed offline. It advances each of
  the ten legacy child cursors from its partially consumed current Parquet file
  to the next untouched file, so restart does not reread multi-GiB partial files.
- The already committed portion of every partial file remains in the local
  corpus. Only each file's unconsumed tail is omitted; later untouched files from
  the same FineWeb source replace those records toward the 54B-token target.
- The original pickle is retained as `cursor-XXXXXXXX.legacy-multistream.pkl`
  before the active cursor is atomically converted.
- `ASTERLM_LEGACY_RESUME_POLICY=exact` remains available for forensic exact
  recovery, but it can reread several GiB per partial file before token progress
  advances and is not recommended for this campaign.
- Every progress bar displays process RSS, system available RAM, and stream
  layout.
- On a 32 GiB host, a 20 GiB process RSS ceiling is recommended; a
  system-available-memory floor is also enforced.
- A memory-pressure stop commits the cursor and local shard, exits with code 75,
  and is not automatically relaunched by `download_data.py`.

After the offline conversion, checkpoint 208 should report and print:

```text
cursor_kind:      asterlm-next-shard-migration
stream_layout:    legacy-multistream-next-shard-v1
partial_files_skipped:10
```

After the first successful checkpoint, the cursor remains an AsterLM-owned
sequential next-shard cursor. Brand-new sources print:

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

## v4: bounded shutdown and broken-IPv6 protection

A real Fedora diagnostic showed the materializer sleeping in `poll()` with a single
TLS socket stuck in IPv6 `SYN-SENT`, while `download_data.py` slept on its stdout
pipe. There was no cache lock and no CPU work. The old shell runner launched
`download_data.py` as a background job and sent SIGINT only to that parent PID;
background SIGINT semantics plus the still-running materializer meant Ctrl-C could
print a shutdown message and then wait forever.

`RUN_100B_SAFE.sh` now execs a foreground Python supervisor. The supervisor starts
`download_data.py` and every materializer in a dedicated process group, forwards
shutdown to the complete group, waits a finite grace period, then escalates from
SIGINT to SIGTERM and finally SIGKILL. This guarantees that closing a terminal is
not required.

The supervisor also enables IPv4-only DNS selection for its Python descendants by
default. This is process-local and does not modify Fedora networking. Disable it
only on a known-good IPv6 network with `./RUN_100B_SAFE.sh --allow-ipv6`.
Explicit Hugging Face timeout environment values now override network-profile
defaults; the safe runner uses 120-second download/connect inactivity and
30-second metadata bounds while retaining two Xet range requests.

## v5: zero-replay offline migration for checkpoint 208

The exact legacy cursor stores row counts inside ten current Parquet files, not
byte offsets or row-group identifiers. Hugging Face exact restoration must open
each file from its beginning and scan to roughly row 414,000 before the first new
record can be yielded. On the observed FineWeb files this consumed about 2.57 GiB
with no visible token progress and would repeat across additional child streams.

Convert the cursor while every downloader process is stopped:

```bash
python scripts/migrate_legacy_cursor.py \
  data/corpus-frontier-16b/fineweb_edu --dry-run

python scripts/migrate_legacy_cursor.py \
  data/corpus-frontier-16b/fineweb_edu
```

This operation is local-only and does not open Hugging Face or download bytes. It
keeps checkpoint ID 208, all 4,847,909,935 committed tokens, every finalized local
shard, and a backup of the original cursor. Verify it with:

```bash
python scripts/inspect_download_cursor.py \
  data/corpus-frontier-16b/fineweb_edu
```

Then run `./RUN_100B_SAFE.sh`. The first network bytes should belong to the next
untouched source file and token progress should begin after that file's first
small Arrow batch is decoded.

To restore the original exact cursor before a newer checkpoint is committed:

```bash
python scripts/migrate_legacy_cursor.py \
  data/corpus-frontier-16b/fineweb_edu --restore
```

## v6: upgrading an already-sequential v3/v4 cursor

Some interrupted exact-resume attempts saved an `asterlm-hf-sequential-v1` cursor
before the offline next-shard policy was introduced. Such cursors have no
`migration_policy` field and still point inside large partial Parquet files.
`migrate_legacy_cursor.py` now upgrades those cursors offline as well. It backs
up the active cursor as `cursor-XXXXXXXX.pre-next-shard.pkl`, advances only
children that are actually partway through a shard, and leaves children already
at an untouched shard boundary unchanged.

## v7: restart policy and throughput recovery

The next-shard policy is now persistent rather than a one-time migration. A
legacy FineWeb checkpoint can land inside another large Parquet file after making
new progress. On every subsequent restart AsterLM inspects the committed child
cursor locally and advances only newly-partial legacy files to the next untouched
file before any remote read. This avoids recurring multi-GiB row replay.

Throughput and decoded-data concurrency are intentionally separated:

- `max_buffer_input_shards=1` remains the Arrow-memory safety boundary. It
  prevents `datasets==5.x` from opening ten decoded input streams and building
  the 30 GiB backlog observed on the laptop.
- `--network-mode balanced` uses 8 Xet range GETs per remote file and is the new
  safe-runner default.
- `--network-mode safe-fast` uses 16 Xet range GETs per file without enabling
  `HF_XET_HIGH_PERFORMANCE`; this is the recommended faster setting on the
  32 GiB RTX 4080 laptop when balanced mode leaves bandwidth unused.
- `--network-mode fast` still enables Hugging Face high-performance mode. It is
  deliberately not the default because it can use substantially more CPU,
  network, disk I/O and buffering.

The important distinction is that Xet range concurrency transfers byte ranges
for one active file; it does not restore the ten concurrently decoded Arrow
input streams that caused the original RAM failure.
