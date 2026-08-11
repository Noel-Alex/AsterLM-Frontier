# Data acquisition, cleaning and curriculum

## Principle

At sub-2B scale, data quality, repetition, curriculum and post-training often matter more than an additional architectural novelty. The final model should not be trained directly from unversioned streaming endpoints. Sources are first validated, materialized, cleaned, deduplicated, decontaminated and audited.

## Full pretraining corpus

Active configuration: `configs/corpus/corpus_frontier_16b.yaml`. Stack-Edu is retired and is not scheduled by any active download profile.

| Source | Target tokens | Weight before adaptive rebalancing | Role |
|---|---:|---:|---|
| FineWeb-Edu-Dedup | 10.4B | 56.5% | broad educational web base |
| DCLM baseline | 2.4B | 13.0% | diverse high-quality web distribution |
| Cosmopedia-v2 | 1.4B | 7.6% | structured synthetic explanations/textbook prose |
| FineMath-4+ | 1.8B | 9.8% | mathematical notation and reasoning |
| Replacement code candidate | not promoted | pending | code generation, syntax, FIM and software knowledge |

The active historical frontier tranche targets 16B tokens before cleaning. Larger 43.5B and 87B prose/math tranches exist, but none is called a complete 50B/100B mixture until a replacement math/code candidate passes the promotion gate. Cleaning and deduplication will reduce usable tokens. The trainer reports effective epochs/repetition after cleaning.

## Retired Stack-Edu source

Stack-Edu rows reference Software Heritage blobs rather than containing code text inline. Its historical materializer and configs are retained only for reproducibility. The source is permanently excluded from active acquisition, cleaning and training plans because the per-blob reconstruction path was not operationally viable for this campaign. Existing shards remain visible as provenance and must not be mixed.

The replacement code tranche must provide direct, resumable materialization and pass license, quality, duplication, benchmark-contamination and matched learning-curve gates. FIM is enabled only after promotion.

The retired materializer historically:

- downloads blobs in parallel,
- retains only rows marked permissive in source metadata,
- preserves repository/blob/license/provenance fields,
- writes resumable compressed shards,
- supports language-specific token targets.

Language targets:

- Python 760M
- C++ 320M
- JavaScript 220M
- TypeScript 160M
- Java 260M
- SQL 180M
- Rust 180M
- Go 140M
- Shell 90M
- C# 90M

License metadata must still be audited before redistributing weights; “permissive” source labels are not a substitute for legal review.

## Immediate download commands

For a slow connection, authenticate and run the low-bandwidth profile:

```bash
hf auth login
python scripts/data_preflight.py --minimum-free-gib 90
python scripts/download_data.py --profile all --validate-first --network-mode low
```

or individually:

```bash
python scripts/download_data.py --profile benchmarks --validate-first --network-mode low
python scripts/download_data.py --profile pilot --validate-first --network-mode low
python scripts/download_data.py --profile frontier --validate-first --network-mode low
python scripts/download_data.py --profile posttrain --validate-first --network-mode low
```

`validate_data_sources.py` resolves current revisions and tests config, split and required fields before long downloads. It emits close split-name suggestions when a configured split moved or was misspelled. DCLM uses `mlfoundations/dclm-baseline-1.0-parquet`, avoiding the original `.jsonl.zstd` loader failure while retaining equivalent source content.

The corpus and record materializers persist Hugging Face `IterableDataset.state_dict()` cursors. Each checkpoint commits the source cursor, finalizes a checksummed Zstandard frame, then atomically advances `state.json`. A restarted process removes orphan partial output and resumes from the last committed remote shard/row. Pilot and frontier web/math sources use the same output directory and deterministic source-shard order, so the pilot is a retained prefix of the full materialization rather than a duplicate.

Use `python scripts/download_status.py` for progress and `python scripts/verify_data_shards.py --only-last` for periodic integrity checks. Full low-bandwidth operations are documented in [LOW_BANDWIDTH_DOWNLOADS.md](LOW_BANDWIDTH_DOWNLOADS.md).

## Cleaning pipeline

`clean_corpus.py` and `prepare_frontier_data.py` implement:

All source families share one exact/near-duplicate index during preparation, so duplication is removed across sources rather than only within each source. The completed derivative is sealed by `data/clean-frontier/clean_manifest.json`, which records the canonical data config, cleaning guarantees, source reports, every shard size, and every shard SHA-256. Decision-grade/final training refuses an unsealed corpus or a config whose source paths no longer match that manifest.

### Normalization

- Unicode normalization
- line-ending and whitespace normalization
- control-character handling
- stable text hashes

### Quality filtering

Recorded metrics include:

- character and word counts
- alphabetic fraction
- digit fraction
- punctuation fraction
- line/word repetition
- average word length
- entropy
- URL/code-like patterns

Thresholds are conservative and source-aware rather than assuming prose heuristics fit code and math.

### PII and secrets

The pipeline detects common:

- email/phone/address-like patterns
- API keys and tokens
- private-key blocks
- cloud credentials
- high-risk secret formats

Policies can redact or drop. Reports preserve counts, never raw matched secrets.

### Deduplication

- exact normalized-text hashes
- persistent SQLite index
- SimHash near-duplicate search
- source-aware rejection accounting

A persistent index avoids holding the entire corpus in RAM and allows resumable processing.

### Benchmark decontamination

Benchmark examples are materialized first. A normalized 13-token n-gram index is built and candidate documents with suspicious overlap are rejected/reported. Included benchmark sources cover MMLU, ARC, GSM8K, HellaSwag, TruthfulQA, HumanEval and MBPP.

Decontamination is imperfect: transformed or semantically equivalent solutions can evade n-gram matching. Evaluation results must still be interpreted cautiously.

## Tokenizer

Default tokenizer:

- byte-level BPE
- 32,768 vocabulary
- NFKC normalization
- byte fallback
- explicit chat/control tokens
- fill-in-the-middle prefix/middle/suffix tokens

Train it on a balanced sample from every major domain, not only web prose:

```bash
python scripts/train_tokenizer.py \
  --data data/clean-frontier/pretrain_data.yaml \
  --documents 2000000 \
  --vocab-size 32768 \
  --output artifacts/tokenizer.json
```

Before final training, report token fertility separately for English prose, math, Python, C++, JavaScript, SQL and multilingual SFT data.

## Packing

The packer:

- adds EOS between documents,
- overlaps consecutive blocks by one token so no valid next-token transition is lost,
- masks the artificial transition from one document’s EOS into the unrelated next document,
- supports FIM transformation for code,
- supports response-only SFT labels.

## Post-training corpus

Configuration: `configs/corpus/posttrain_frontier.yaml`.

- Smol-SmolTalk: broad small-model instruction data
- selected SmolTalk2 no-think/multilingual/science/reasoning splits
- OpenThoughts: filtered reasoning material
- UltraFeedback binarized: chosen/rejected preferences

Recommended order:

1. general instruction SFT,
2. verified math/code/reasoning continuation,
3. smaller long-context/tool/multilingual SFT,
4. offline-reference DPO.

Reasoning traces should be quality-filtered. Do not assume more hidden chain-of-thought text always improves a small model.

## Adaptive mixture decisions

Every validation record should include source identity. Track held-out loss by source. Rebalance only when evidence shows a domain is underlearned or over-repeated.

Recommended signals:

- web/math/code validation loss
- token repetition/effective epochs
- GSM8K and math exact match
- HumanEval/MBPP pass@1
- knowledge/commonsense tasks
- natural long-document perplexity
- multilingual regression

Do not select the final mix using one aggregate loss alone.


## Reasoning-model extension

The `reasoning` download profile adds three independently resumable sources under `data/reasoning-frontier`: Mixture-of-Thoughts for cold-start traces, DAPO-Math for exact-answer RL prompts, and the decontaminated tested Python set for executable-code RL. The `all` profile includes these stages without invalidating existing corpus checkpoints. Full preparation and training instructions are in [REASONING_MODEL.md](REASONING_MODEL.md).

## Progressive 50B and 100B overtraining tiers

The repository now supports progressive `overtrain50` and `overtrain100` profiles that expand the same pilot/frontier checkpoints. Use the single-command campaign wrapper:

```bash
python scripts/data_campaign.py --tier 100b --network-mode low
```

The complete rationale, source totals, disk policy, cleaning path and training gates are documented in [100B_EXPERIMENT.md](100B_EXPERIMENT.md).
