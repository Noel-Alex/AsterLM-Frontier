# AsterLM Frontier

AsterLM Frontier is a consumer-GPU language-model research system built around a
12 GiB RTX 4080 Laptop GPU, with resumable burst execution planned for qualified
cloud GPUs. It includes model architecture, pretraining, exact recovery, public
Hugging Face checkpoint durability, W&B telemetry, long-context evaluation,
provider contracts, inference tools, and a local research Studio.

The selected pretraining incumbent is **Aster K3 Stable LatentMoE**:

- **1,448,120,880 total parameters / 568,155,376 active per token**
- 32 layers: 24 Kimi Delta Attention and 8 latent-attention anchors
- 16 routed top-2 experts plus one shared expert
- Stable LatentMoE, SiTU-GLU, sigmoid routing, and quantile balancing
- BF16 laptop compute, packed CUTLASS experts, FLA KDA, and GPU-resident
  blockwise-INT8 Muon state
- bounded 8K latent-attention history for 256K inference validation, with 1M as
  a validation-gated stretch target rather than a quality claim

The final 4K full-gradient laptop gate measured **2,016.5 tok/s, 97% median GPU
utilization, and 8.065 GiB peak allocated VRAM**. That fixes the earlier MoE
underutilization problem at the selected geometry; it does not by itself prove
final language quality.

The 100B curriculum is frozen provisionally at 92B tokens at 4K, then 3B at 8K,
3B at 16K, and 2B at 32K. The run cannot start while mandatory promotion gates
remain red. Current blockers are the two-seed final-scale time-to-quality proof
and the fully cleaned, hashed, audited corpus seal. A bounded GDN2 mixer
challenger and a strong final-scale dense control are the last planned
architecture falsification tests before closing selection.

Read [the canonical current state](docs/CURRENT_STATE.md), then
[START_HERE.md](START_HERE.md) and the [100B experiment contract](docs/100B_EXPERIMENT.md).

## Studio

On Windows:

```powershell
.\ASTER_STUDIO.ps1
```

Studio stays on **http://127.0.0.1:8765** and uses the restored cream theme. It
indexes experiments, findings, run telemetry, promotion gates, corpus state,
checkpoint/provider readiness, and comparisons. Credentials remain in
provider-native stores; Studio stores aliases and readiness only.

## Durability policy

- Checkpoints are public at
  [philoweeb/AsterLM-Frontier-100B](https://huggingface.co/philoweeb/AsterLM-Frontier-100B).
- Cloud checkpoints target roughly five-minute recovery points and are uploaded
  with model, optimizer, scheduler, RNG, data cursor, manifests, and hashes.
- Resume chooses the newest compatible complete checkpoint across local storage
  and the Hub; model-only recovery is rejected for final training.
- Local checkpoint storage is bounded at 150 GiB. Hub guards warn at 7.0 TB and
  stop at 7.5 TB.
- W&B, JSONL, TensorBoard, diagnostic bundles, and the Studio research archive
  preserve both positive and negative results.

## Honest boundaries

- No trained 1.448B checkpoint exists yet; the current inference canary proves
  runtime compatibility with random initialization, not model capability.
- KDA is not DeepSeek CSA/HCA. CSA/HCA, mHC, sparse gather attention, MTP, FP8,
  and other frontier mechanisms remain separately evidence-gated challengers.
- The laptop path is real; Modal/GCP and each cloud GPU/backend combination are
  implemented as contracts but are not qualified until explicit, budgeted tests
  pass. No paid cloud execution is authorized by repository readiness alone.
- 100B tokens is a research ceiling and trajectory, not a guarantee of grokking
  or frontier-lab capability.

## Repository map

```text
configs/              selected and challenger model/train/data contracts
src/asterlm/          model, kernels, optimizer, data, checkpoint and runtime code
scripts/              campaigns, evidence importers, cloud adapters and tooling
studio/               fixed-port local research/control UI
docs/promotion-evidence/  compact hashed evidence required by clean-clone CI
docs/research/        append-only findings and architecture records
tests/                CPU, CUDA, contract and recovery regression tests
```

Post-training is intentionally deferred until base pretraining is launched; its
existing SFT/preference/RLVR machinery remains available but is not the current
critical path.
