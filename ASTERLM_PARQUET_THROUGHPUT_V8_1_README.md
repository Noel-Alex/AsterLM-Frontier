# AsterLM Parquet Throughput v8.1

This replaces the malformed v8 unified diff.

If `git apply` reported `patch with only garbage`, that malformed patch did not
change the repository.

## Apply

Extract into the AsterLM repo root and run:

```bash
cd ~/Documents/AsterLM-Frontier
source .venv/bin/activate

unzip -o ~/Downloads/AsterLM-parquet-throughput-v8.1.zip -d .
python APPLY_PARQUET_THROUGHPUT_V8_1.py
```

The installer validates all expected source snippets before changing anything,
backs up both modified scripts, is idempotent, and runs `compileall`.

## Run

```bash
export HF_XET_FIXED_DOWNLOAD_CONCURRENCY=24
export HF_XET_CLIENT_MAX_IDLE_CONNECTIONS=32
unset HF_XET_HIGH_PERFORMANCE

./RUN_100B_SAFE.sh --network-mode safe-fast
```

Expected new startup lines:

```text
Arrow I/O threads:        24
Parquet prefetch ranges:  8
Parquet range size:       128 MiB
```

## More aggressive tuning

```bash
export ASTERLM_ARROW_IO_THREADS=32
export ASTERLM_PARQUET_PREFETCH_LIMIT=12
export ASTERLM_PARQUET_RANGE_MIB=192
./RUN_100B_SAFE.sh --network-mode safe-fast
```

## Disable prefetch tuning without reverting

```bash
export ASTERLM_PARQUET_PREFETCH=0
```

## Restore backups

```bash
cp .asterlm-patch-backup-v8.1/materialize_corpus.py scripts/materialize_corpus.py
cp .asterlm-patch-backup-v8.1/run_100b_safe.py scripts/run_100b_safe.py
```
