# Start here

AsterLM Frontier is an experiment-driven LLM laboratory for a 12 GiB RTX 4080 Laptop GPU. The repository supports Windows as the control plane and Linux/WSL as the CUDA execution environment. Architecture candidates are promoted only after matched quality, throughput, memory, stability, and inference tests.

## Open the Studio

From Windows, run:

```powershell
.\ASTER_STUDIO.ps1
```

or double-click `ASTER_STUDIO.cmd`. The restored cream Studio theme includes run monitoring, experiment evidence, and provider/profile readiness. Provider secrets stay in provider-native stores and are represented in Studio only by aliases and readiness state.

## Current CUDA environments

- Fedora target: system CUDA 13.1, preferred automatically when `/usr/local/cuda-13.1` exists.
- This WSL validation environment: PyTorch `2.13.0+cu130` plus NVIDIA CUDA 13.3 compiler wheels.
- Validated on the RTX 4080 Laptop GPU: Transformer Engine 2.17 FP8 (delayed and current scaling), grouped MoE, FLA 0.5.2 AttnRes, FlexAttention, and fused linear cross entropy.

Capture the exact environment before a consequential run:

```bash
python scripts/capture_runtime_manifest.py --output runs/setup/runtime-manifest.json
python scripts/frontier_vnext_capabilities.py --json runs/setup/capabilities.json
```

Never compare or resume runs whose manifests silently disagree on code, model/train/data configs, attention backend, or runtime.

## Current data state

The old 500M note is obsolete. The on-disk 100B-tier state currently records:

| Source | Materialized | Planned | State |
|---|---:|---:|---|
| FineWeb-Edu | 54.000B | 54B | complete |
| DCLM | 16.000B | 16B | complete |
| Cosmopedia-v2 | 6.000B | 6B | complete |
| FineMath-4+ | 8.032B | 11B | source exhausted |
| Stack-Edu historical shard | excluded | 0 | retired; provenance only |
| Replacement math/code tranche | 0 | at least 15.968B | candidate audit required |

That is about **84.032B active materialized pretraining tokens**, not yet a complete or cleaned 100B training mixture. FineMath cannot supply its remaining 2.968B target from the pinned source, and the code replacement is not promoted. Do not launch the final campaign until replacement/additional math and code sources have been selected, licensed, downloaded, cleaned, deduplicated, decontaminated, and mixed deliberately.

Inspect the live state instead of relying on this snapshot:

```bash
python scripts/download_status.py --json
python scripts/verify_data_shards.py data/corpus-frontier-16b --only-last
```

All benchmark/decontamination splits currently recorded on disk are complete: ARC-Challenge, GSM8K, HellaSwag, HumanEval, MBPP, MMLU, and TruthfulQA.

## Current architecture decision

There is no accepted final architecture yet. Dense all-MLA is a control, and the KDA/MoE candidates remain experimental. The current matched utilization follow-up confirms that the small single-GPU MoE workloads underfeed the GPU; balanced routing means expert collapse is not the explanation. Dense-KDA controls and operator-level profiles are required before changing kernels or architecture.

Read these first:

- [engineering and research handoff](docs/HANDOFF_2026-08-10.md)
- [vNext2 findings](docs/experiments/frontier-vnext2/FINDINGS_2026-08-10.md)
- [kernel compatibility boundaries](docs/research/KERNEL_COMPATIBILITY_2026-08-10.md)
- [100B experiment design](docs/100B_EXPERIMENT.md)
- [NVIDIA Nemotron candidate pool](docs/data/NEMOTRON_CANDIDATE_POOL_2026-08-10.md)
- [training runbook](docs/TRAINING_RUNBOOK.md)
- [environment setup](docs/ENVIRONMENT.md)

Official FlashKDA and DeepSeek FlashMLA kernels do not support this laptop's Ada SM89 target in their current published builds. Aster therefore uses FLA's portable Triton KDA training path while it measures alternatives. Do not install an incompatible kernel merely because its paper or datacenter benchmark is impressive.

## Before any expensive run

1. Confirm a clean or intentionally recorded Git state.
2. Capture runtime, configuration hashes, dataset state, and GPU baseline.
3. Run the training preflight and a short checkpoint/resume smoke test.
4. Define success and stop criteria before spending compute.
5. Upload checkpoints and manifests atomically; keep W&B/Hugging Face run identities in the local registry.
6. Treat 100B tokens and “grokking” as research hypotheses, not guarantees. Scale only after smaller controlled runs show useful quality-per-token and throughput-per-cost.

The detailed command set for pretraining, resume, evaluation, inference, export, SFT, preference training, and verifier-backed reasoning training is in the [training runbook](docs/TRAINING_RUNBOOK.md).
