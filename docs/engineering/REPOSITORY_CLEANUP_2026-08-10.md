# Repository cleanup record - 2026-08-10

## Safety baseline

- Cleanup branch: `Noel/frontier-hardening`
- Preserved source commit: `a772cbd76c253c3fd86c7e2ea569ea9b86b749c7`
- Safety branch: `Noel/pre-cleanup-a772cbd`
- The working tree contained no uncommitted source changes. Apparent changes after the
  Fedora-to-NTFS copy were executable-bit and symlink representation differences.
- `core.filemode=false`, `core.symlinks=false`, and the repository `.gitattributes`
  make the shared checkout stable for Windows Git and WSL. The native ext4 runtime
  clone restores real Unix symlinks.

## Removed point-in-time backup trees

The directories below were already tracked by Git. They are being removed only from
the current tree; every byte remains recoverable from its recorded source commit or
the safety branch.

| Backup tree | Tracked files | Recovery commit |
|---|---:|---|
| `.asterlm-patch-backup-v8.1` | 2 | `34da84e` |
| `.asterlm-backup-turbo-v9-20260807-225226` | 5 | `34da84e` |
| `.asterlm-backup-turbo-v10-20260807-231710` | 5 | `9ddaba5` |
| `.asterlm-backup-v10.1-20260808-074609` | 2 | `9ddaba5` |
| `.asterlm-backup-v10.1-20260808-074627` | 2 | `9ddaba5` |
| `.asterlm-backup-v10.1b-20260808-075245` | 2 | `ac8245c` |
| `.asterlm-backup-downloader-v11-20260808-082051` | 4 | `ac8245c` |
| `.asterlm-backup-downloader-v11.1-20260808-082908` | 5 | `ac8245c` |
| `.asterlm-backup-deterministic-profiler-20260809-093404` | 1 | `5dccb67` |
| `.asterlm-backup-fp8-moe-20260809-004458` | 1 | `5dccb67` |
| `.asterlm-backup-fp8-safety-20260809-084007` | 2 | `5dccb67` |
| `.asterlm-backup-fp8-safety-v2-20260809-084307` | 2 | `5dccb67` |
| `.asterlm-backup-precision-20260808-232933` | 1 | `5dccb67` |
| `.asterlm-backup-te-checkpoint-20260809-005115` | 1 | `5dccb67` |
| `.asterlm-backup-te-checkpoint-v2-20260809-005357` | 1 | `5dccb67` |
| `.asterlm-backup-te-checkpoint-v2-20260809-005610` | 1 | `5dccb67` |
| `.asterlm-backup-te-grouped-moe-20260809-091320` | 1 | `5dccb67` |
| `.asterlm-backup-te-grouped-moe-v2-20260809-092810` | 1 | `5dccb67` |
| `.asterlm-studio-backup-20260808-225310` | 15 | `5dccb67` |
| `.asterlm-backup-frontier-vnext-20260809T084334Z` | 8 | `a772cbd` |
| `.asterlm-backup-frontier-vnext2-20260809T102739Z` | 6 | `a772cbd` |

Example recovery without changing the working tree:

```bash
git show 5dccb67:.asterlm-backup-fp8-moe-20260809-004458/moe.py
```

## Artifact and data disposition

- `artifacts/tokenizer_proxy.json` is a generated proxy tokenizer. It remains on the
  workstation but is no longer tracked in the source tree.
- `data/`, `runs/`, `artifacts/`, checkpoints, W&B state, profiler traces, and local
  provider credentials are ignored by Git.
- The approximately 116 GiB checkout footprint is overwhelmingly downloaded corpus
  and experiment evidence. It is intentionally retained; it is not cleanup junk.
- Canonical small manifests, corrected ledgers, decisions, and schemas belong under
  `docs/` or `research/` so evidence remains reviewable in Git without committing
  mutable checkpoints or raw datasets.

## Removed legacy delivery files

The root-level zip-patch apply scripts, patch manifests, stale generated status/audit files, and a
tracked pre-schema-fix YAML backup were removed in a second cleanup pass. Their recovery commits
and canonical replacements are indexed in `docs/archive/LEGACY_DELIVERY_ARTIFACTS_2026-08-10.md`.

## Removed broken nested-checkout snapshot

`third_party/native-sparse-attention` was a partial tracked copy of a nested Git checkout. Its Unix
symlinks targeted absent submodule directories, while `frontier_vnext2_sparse_scout.py` could not
clone the pinned upstream repository over the existing non-empty tree. The snapshot was removed from
the current tree and remains recoverable from `a772cbd`. The scout now clones commit
`bd67af59b90afa34b25f61d2922e612d10dba3bd` into ignored
`.cache/third_party/native-sparse-attention`, keeping reproducible external source out of AsterLM's
tracked source tree and working correctly from both Windows and native Linux checkouts.
The scout does not initialize NSA's historical FLA, Flame, or TorchTitan submodules and does not
replace AsterLM's tested FLA installation. The pinned NSA source expects FLA 0.4's former
`fla.ops.common.utils` import path; the isolated scout aliases that path in memory to the equivalent
FLA 0.5 `fla.ops.utils` helpers and loads only the NSA kernel package.

## Cross-platform runtime layout

- Shared Windows checkout: `N:\AsterLM-Frontier`
- Native WSL/ext4 execution clone: `/root/work/AsterLM-Frontier`
- Isolated WSL environment: `/root/.venvs/asterlm`

The Windows checkout is the user-visible source of truth. GPU training and profiling
run under WSL from ext4; controlled sync/commit steps keep the two trees aligned.

## Static-quality baseline

A repository-wide Ruff audit found 265 findings: 92 broad-exception warnings, 35 import-order
findings, 25 silent-exception warnings, 16 unused imports, and smaller style categories. Do not apply
a blind whole-tree rewrite; broad exception handling is often deliberate at durable data/run
boundaries and should be reviewed by subsystem. The sole `F823` undefined-local finding was a real
bug in `scripts/studio_prepare_data.py`: a nested `import shutil` made its earlier disk-space preflight
reference a local before assignment. The nested import was removed and the focused `F823` gate is
clean.
