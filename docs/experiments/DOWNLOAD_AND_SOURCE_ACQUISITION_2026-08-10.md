# Download and upstream-source acquisition log

Date: 2026-08-10

This log records the engineering and data acquisition work started during the
architecture campaign. Raw data, caches, and cloned upstream worktrees are ignored
artifacts; their pinned inputs and decisions remain tracked here and in configuration.

## Upstream engineering sources

Shallow checkouts of Megatron-LM, TorchTitan, DeepSpeed, Transformer Engine, Kimi
Linear, TileKernels, DeepGEMM, FlashMLA, MegaBlocks, SGLang, and vLLM were placed
under `/root/.cache/asterlm/upstreams`. Exact repository URLs, commits, and intended
uses are locked in `configs/research/upstream_sources_2026-08-10.yaml`.

Downloading a source tree does not promote its runtime. Hardware compatibility,
numerical parity, checkpoint recovery, quality, and sustained throughput remain
mandatory gates.

## Dataset launches

### NVIDIA Nemotron candidate pool

The Hugging Face account was authenticated and preflight passed package, codec,
cursor, cache-writability, and disk checks with approximately 678 GiB free.

- `nvidia/Nemotron-CC-Math-v1`, `4plus`, pinned revision
  `397a2502f2028c659ba411a6c4935b464a7f03aa`, passed config/split/schema sampling.
  Its 3B-token materialization started under
  `data/corpus-nemotron-candidates/nemotron_cc_math_4plus`.
- The automatic 3.7 GiB process-RSS guard stopped safely at approximately 414M
  tokens when one decoded Arrow row group reached 4.0 GiB. The cursor and output
  shard were committed. The run resumed with a still-bounded 6 GiB process ceiling
  and the independent 2 GiB available-memory floor intact.
- The resumed Math materializer completed successfully at 3,000,000,834 estimated
  tokens, checkpoint 167, with `last_checkpoint_reason: target`. Its download
  launcher exited zero and retained the run manifest and per-stage logs.
- `nvidia/Nemotron-CC-Code-v1` remains gated for the authenticated account. Its
  license/access terms must be accepted by the user; authorization failures are not
  retried as network failures.
- `nvidia/Nemotron-Pretraining-Code-v1` has not been started pending its independent
  access validation. Neither code source is represented as downloaded.

These are candidate inputs, not an automatic addition to the final 100B mixture.
Cleaning, cross-corpus deduplication, decontamination, license review, held-out
validation, and matched data-mixture ablations still precede promotion.

### Stack-Edu code throughput repair

The retained Stack-Edu Python cursor began at about 53.9M of its 760M-token target.
The original `workers: 4`, `in_flight: 16` policy produced highly variable roughly
5--15k tok/s because each metadata record requires a separate Software Heritage S3
object request. CPU and memory remained mostly idle.

The process was interrupted once through SIGINT, drained its pending requests, and
published checkpoint 39 at 54,970,909 estimated tokens with reason
`keyboard_interrupt`. It then resumed from that cursor with 32 workers and 128
in-flight requests. Early warmed observations were commonly 38--56k tok/s, with
occasional higher bursts, at roughly 0.8--0.9 GiB RSS. Sustained completion-rate and
S3-throttling evidence are still required; early bursts are not a final benchmark.

The one-shard Arrow input bound remains unchanged. Only independent blob-fetch
latency is overlapped.

On the user's direction, Stack-Edu was dropped from the active acquisition plan
after a final clean SIGINT checkpoint at 83,969,133 Python tokens. The
partial materialization is retained as a recoverable artifact but is excluded from
the intended training mixture and will not be resumed unless explicitly requested.
The short tuning exercise established that 64 workers could sustain about 41k
tok/s over a warmed committed interval, compared with roughly 36k tok/s at 32
workers and the original roughly 5--15k tok/s at 4 workers; this result is retained
only as downloader evidence, not as a reason to promote the dataset.

### Reasoning and post-training

The reasoning profile initially failed before writing DAPO rows because a stale
shared Hugging Face builder cache exposed only a `default` config. An exact clean
load of the pinned source accepted the configured `all` split. Rather than deleting
cache entries used by active downloads, the reasoning profile restarted with the
isolated cache `data/hf-cache-reasoning-v2`.

- DAPO Math completed 17,398 records.
- The verifiable Python stage began and retained its independent cursor/shards.
- Mixture-of-Thoughts follows in the same resumable profile.
- The post-training profile began with SmolTalk and retained its own source cursors.

The decontamination benchmark corpus was already present and was not downloaded a
second time.

## Reliability fix

`is_retryable_exception` now classifies gated-dataset, ask-for-access, unauthorized,
and authentication-required failures as permanent. Retrying a license/access denial
cannot grant access and previously wasted exponential-backoff time.
