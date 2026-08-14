# Code tranche decision — 2026-08-13

## Status

The final pretraining corpus is **not sealed**. The machine has 87,031,563,044
materialized prose/math tokens, but the intended 13B-token code tranche still
requires a deliberate source and license decision. No replacement dataset is
counted merely because its repository exists or can be streamed.

### Access recheck — 2026-08-14

The local Hugging Face credential is valid for account `philoweeb`. A read-only
Hub audit enumerated 167 Parquet files in `Nemotron-CC-Code-v1` and 316 matching
`Synthetic-Code` Parquet files in `Nemotron-Pretraining-Code-v1`, then requested
metadata for one real data object from each repository. Both object requests
returned HTTP 403 gated-access errors. This verifies that the blocker is the two
human dataset agreements, not authentication, repository discovery, tooling, or
an accidental download failure. No dataset bytes were downloaded during the
check.

## Preferred source

The preferred candidate remains:

- 8B sampled tokens from
  [`nvidia/Nemotron-CC-Code-v1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1);
- 5B sampled tokens from the `Synthetic-Code` configuration of
  [`nvidia/Nemotron-Pretraining-Code-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1).

The August 13 Hub audit found both repositories public but manually gated. The
authenticated `philoweeb` profile can read repository metadata but has not been
granted the data files. `Nemotron-CC-Code-v1` reports 427.9B source tokens and
directly materialized Parquet text; its repository is about 563GB, so Aster must
stream only the pinned, token-bounded sample rather than clone it. NVIDIA describes
the corpus as code-bearing Common Crawl pages filtered by a Lynx/LLM pipeline and a
three-level quality classifier.

`Nemotron-Pretraining-Code-v1` is about 246GB at the repository level and exposes a
direct `Synthetic-Code` configuration. Its data agreement allows model training,
but the card warns that some synthetic subsets can impose redistribution/use terms
from the generating Qwen, DeepSeek, or Phi models. Agreement acceptance and a final
license review are therefore human gates. The automation must not accept them on the
user's behalf.

## Rejected direct substitute

`HuggingFaceTB/smollm-corpus`, configuration `python-edu`, is not direct training
text. Dataset Viewer reports 7,678,448 rows and roughly 644MB of Parquet, but the
rows contain repository/blob identifiers, paths, sizes, and scores. Reconstructing
the actual GitHub blobs is the slow Stack-Edu path the user explicitly retired.

## Audited fallback

If NVIDIA access is declined, the most practical directly materialized fallback
found in this pass is
[`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean).
Dataset Viewer reports 11,027,000 rows, about 32GB of generated Parquet and about
95GB uncompressed in memory. It is ungated and provides language-plus-license
configurations, allowing Aster to restrict ingestion to permissive families such
as Apache-2.0, MIT, BSD, ISC, CC0, and Unlicense.

This fallback is not an automatic promotion. Its card documents only basic line
length, alphanumeric-fraction, and generated-file filters. Aster would still need
to perform license/config selection, exact and near deduplication, secret/PII
handling, benchmark decontamination, language balancing, FIM validation, and a
matched mixture ablation. The older gated StarCoder data is more heavily processed
but carries original-license/attribution obligations and is not preferred over the
newer NVIDIA candidate.

## Promotion rule

1. Ask the user to review and accept the two NVIDIA agreements in the browser.
2. Re-run authenticated file access without downloading the corpus.
3. If access works, materialize exactly the declared 8B + 5B token bounds with
   pinned revisions and resumable direct-Parquet cursors.
4. If access is declined, materialize a license-restricted CodeParrot candidate.
5. Clean it together with all prose/math sources using the shared global dedup DB.
6. Run two-seed small mixture comparisons for general, math, and code validation.
7. Only the winning audited mixture may produce the final clean manifest and
   tokenizer seal.

No 92B-token Stage 1 launch may treat `expected_materialized_tokens_after_downloads`
as actual evidence.
