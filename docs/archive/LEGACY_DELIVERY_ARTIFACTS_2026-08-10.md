# Legacy delivery artifacts removed from the working tree

Date: 2026-08-10

Early AsterLM iterations were transferred as zip patches. Their apply scripts, manifests, backup
copies, and point-in-time reports remained in the repository root after the changes had already
been incorporated into canonical source and documentation. They were removed to make the checkout
an ordinary source repository rather than a stack of patch deliveries.

| Removed artifact | Last containing commit | Current source of truth |
|---|---|---|
| `APPLY_DOWNLOAD_RESUME_FIX.sh` | `615b8e3` | `scripts/materialize_corpus.py`, `src/asterlm/data/`, `docs/DOWNLOAD_RESUME_MEMORY_FIX.md` |
| `APPLY_PARQUET_THROUGHPUT_V8_1.py` | `34da84e` | `scripts/materialize_corpus.py`, `scripts/run_100b_safe.py` |
| `ASTERLM_PARQUET_THROUGHPUT_V8_1_README.md` | `34da84e` | `docs/LOW_BANDWIDTH_DOWNLOADS.md` |
| `PATCH_BASE_COMMIT.txt` | `615b8e3` | Git history and the download-resume document |
| `PATCH_MANIFEST.txt` | `615b8e3` | Git tree |
| `PATCH_MANIFEST_V6.txt` | `615b8e3` | Git tree |
| `PATCH_MANIFEST_V7.txt` | `615b8e3` | Git tree |
| `REASONING_PATCH_MANIFEST.txt` | `6565562` | `docs/REASONING_MODEL.md`, `docs/RLVR_IMPLEMENTATION.md` |
| `REASONING_PATCH_NOTES.md` | `6565562` | Current reasoning documents and runbook |
| `DOWNLOAD_FIX.md` | `6565562` | Current download documents and runbook |
| `PROJECT_STATUS.md` | `6565562` | Current experiment findings, decisions, and eventual handoff |
| `AUDIT_REPORT.md` | `6565562` | Current audit/findings documents and test evidence |
| `configs/corpus/posttrain_frontier.yaml.before-schema-fix.bak` | `6565562` | `configs/corpus/posttrain_frontier.yaml` |

The historical files can be inspected without changing the working tree, for example:

```bash
git show 615b8e3:APPLY_DOWNLOAD_RESUME_FIX.sh
git show 6565562:AUDIT_REPORT.md
```

The cleanup safety branch `Noel/pre-cleanup-a772cbd` also preserves the complete pre-cleanup tree.
