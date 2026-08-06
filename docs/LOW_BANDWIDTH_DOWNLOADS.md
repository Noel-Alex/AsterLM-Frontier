# Reliable downloads on a slow connection

The data materializers are designed to be rerun for days or weeks. The authoritative progress files are the per-source `state.json` and `cursor-*.pkl` files under `data/`. Hugging Face cache files are useful but are not the source of truth for the materialized corpus.

## One-time environment repair

Activate the repository environment, upgrade the Hub stack, and reinstall the editable project:

```bash
source .venv/bin/activate
python -m pip install --upgrade \
  "huggingface_hub>=0.32.0" \
  "datasets>=3.5" \
  "zstandard>=0.23"
python -m pip install -e ".[cuda,dev]"
```

Authenticate even for public datasets. Public anonymous access works, but an authenticated token receives higher Hub limits:

```bash
hf auth login
```

The token is stored by Hugging Face. Do not place it in repository YAML or logs.

Run the local preflight:

```bash
python scripts/data_preflight.py --minimum-free-gib 90
```

It checks imports, the fsspec Zstandard registry, Hugging Face authentication, writable cache storage, disk capacity, and support for shard-aware `IterableDataset.state_dict()` checkpoints.

## Recommended long-running command

The low-bandwidth mode raises HTTP timeouts and limits Xet range concurrency rather than attempting to saturate the connection:

```bash
python scripts/download_data.py \
  --profile all \
  --validate-first \
  --network-mode low
```

`--profile all` is progressive rather than duplicative:

1. The 500M-token pilot is written into `data/corpus-frontier-16b`.
2. The full frontier stage resumes those same source checkpoints and expands them to the larger targets.
3. Benchmarks and post-training data have independent checkpoints.

DCLM uses the official Parquet mirror rather than the `.jsonl.zstd` repository. Parquet avoids the original codec failure and lets the loader request only the columns the materializer keeps.

## Keep it running after closing the terminal

On Fedora, use `tmux` and inhibit automatic sleep while the process is active:

```bash
sudo dnf install -y tmux
tmux new -s aster-download
```

Inside the tmux session:

```bash
source .venv/bin/activate
systemd-inhibit \
  --what=sleep:idle \
  --mode=block \
  --why="AsterLM corpus download" \
  python scripts/download_data.py \
    --profile all \
    --validate-first \
    --network-mode low
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t aster-download
```

## Recovery behavior

The materializers use Hugging Face iterable-dataset cursor checkpoints, not only a record count. The cursor records the current remote shard and row. At every checkpoint:

1. the remote dataset cursor is written,
2. the current Zstandard output shard is finalized with a frame checksum,
3. `state.json` is atomically advanced last.

After an unclean shutdown, any output newer than the last atomic state pointer is removed. This avoids duplicate finalized records. At most one checkpoint interval is replayed after a hard power loss.

Normal network failures are retried with exponential backoff. The default is 50 transient retries per materializer. To retry indefinitely:

```bash
python scripts/download_data.py \
  --profile all \
  --network-mode low \
  --max-retries 0
```

Schema, authentication, missing-disk-space, and unsupported-codec errors are treated as permanent and fail instead of looping forever.

To stop cleanly, press `Ctrl-C` once. The corpus and record materializers save the current cursor and shard. Stack-Edu may wait for already-started Software Heritage requests before committing its cursor.

To continue, rerun the exact same command.

## Monitor progress

```bash
python scripts/download_status.py
```

Machine-readable status:

```bash
python scripts/download_status.py --json > data/download-status.json
```

Per-stage console logs are written to:

```text
data/download-logs/
```

The orchestration manifest is:

```text
data/download_run.json
```

For example:

```bash
tail -f data/download-logs/frontier-corpus-fineweb_edu.log
```

## Integrity checks

Every newly written local shard has a Zstandard frame checksum. Verify only the most recent shard in each source periodically:

```bash
python scripts/verify_data_shards.py --only-last
```

Verify every shard after all downloads finish:

```bash
python scripts/verify_data_shards.py data
```

A failed shard verification should be handled by moving the affected source directory aside and rematerializing it, unless the corresponding checkpoint can be rolled back together with that shard. Never delete only an old finalized shard while retaining a cursor that has already advanced past it.

## Files not to delete during a download

Keep all of these together:

- `data/**/state.json`
- `data/**/cursor-*.pkl`
- finalized `*.jsonl.zst` shards
- `data/hf-cache/`

Files ending in `.partial` are uncommitted. The next run cleans stale partials automatically.

## Useful overrides

Use a different cache disk:

```bash
python scripts/download_data.py \
  --profile all \
  --network-mode low \
  --hf-home /path/on/large/ssd/huggingface
```

Checkpoint every five minutes instead of the default fifteen:

```bash
python scripts/download_data.py \
  --profile all \
  --network-mode low \
  --checkpoint-seconds 300
```

More frequent checkpoints reduce replay after a power loss but create smaller and more numerous local shards.

Use `--network-mode balanced` only after the low mode has proven stable. `fast` enables Hugging Face Xet high-performance mode and is inappropriate for a slow or unstable connection.


## Reasoning-model extension

The `reasoning` download profile adds three independently resumable sources under `data/reasoning-frontier`: Mixture-of-Thoughts for cold-start traces, DAPO-Math for exact-answer RL prompts, and the decontaminated tested Python set for executable-code RL. The `all` profile includes these stages without invalidating existing corpus checkpoints. Full preparation and training instructions are in [REASONING_MODEL.md](REASONING_MODEL.md).
