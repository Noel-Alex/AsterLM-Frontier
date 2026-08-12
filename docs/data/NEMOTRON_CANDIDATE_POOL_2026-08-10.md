# NVIDIA Nemotron candidate pool

Date: 2026-08-10

## Decision

Add a separately materialized, revision-pinned 16B-token candidate pool. Do not silently fold it into the final 100B mixture. The current raw corpus has about 84.032B prose/math tokens and 0.054B code tokens; FineMath-4+ exhausted its pinned source 2.968B below target. The new pool gives the cleaning and ablation stages enough math/code material to close that gap without pretending every downloaded token should be trained on.

| Candidate | Download target | Why it is present |
|---|---:|---|
| `nvidia/Nemotron-CC-Math-v1`, `4plus` | 3B tokens | Replaces the unavailable tail of the 11B FineMath allocation with a higher-quality math candidate |
| `nvidia/Nemotron-CC-Code-v1`, `data` | 8B tokens | Code explanations, documentation, and code-bearing web pages |
| `nvidia/Nemotron-Pretraining-Code-v1`, `Synthetic-Code` | 5B tokens | Natural-language/code and synthetic programming diversity |

NVIDIA reports 52B tokens in the 4plus math subset and describes its pipeline as MinHash-deduplicated and decontaminated against MATH, GSM8K, MMLU, and MMLU-Pro. That does not remove AsterLM's obligation to deduplicate it against the active FineWeb, DCLM, FineMath and Cosmopedia data. See the [official math dataset card](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1).

The Common-Crawl code set contains actual training text. By contrast, `Nemotron-Pretraining-Code-v3` contains 146M metadata rows identifying GitHub repositories, relative paths, languages, and commits; it is not itself a ready-to-train 173B-token text corpus. AsterLM therefore does not treat v3 metadata as downloaded code. See the [official CC-Code card](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) and [official Code-v3 card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3).

## Access and license gate

These repositories are gated. The user must log in to Hugging Face and accept the NVIDIA data agreement on each dataset page. The source cards also disclose upstream-model licensing considerations for synthetic data. Access acceptance and license review are explicit campaign gates, not boxes the code can click for the user.

No Hugging Face token was configured in the WSL environment at the time this plan was written. After accepting the terms, authenticate without putting a token in Git or a command history:

```bash
hf auth login
hf auth whoami
python scripts/download_data.py \
  --profile nemotron-candidates \
  --validate-first \
  --require-auth \
  --network-mode safe-fast \
  --dry-run
```

Remove `--dry-run` only after validation reports `ok` for all three sources and disk capacity is confirmed. The materializer resolves and records revisions, writes atomic cursors and checksummed compressed shards, and resumes without changing the selected transformation.

## Promotion gate

Before any candidate enters the final 100B schedule:

1. Verify shard integrity and provenance.
2. Run secret/PII filtering and license-policy checks.
3. Normalize without damaging equations, Markdown, or code structure.
4. Deduplicate within the source and across every existing corpus.
5. Re-run benchmark decontamination, including code benchmarks.
6. Build held-out source-stratified validation sets.
7. Compare small matched mixtures on loss, math/code evaluations, memorization, and downstream regressions.
8. Record exact accepted token counts and update the 100B budget; do not merely append 16B to a 100B schedule.

The candidate configuration is `configs/corpus/corpus_nemotron_candidates_16b.yaml`.
