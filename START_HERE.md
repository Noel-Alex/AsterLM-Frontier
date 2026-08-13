# Start here

Use [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md) as the authoritative snapshot.
Older dated handoffs and runbooks are historical evidence when they disagree with
it.

## 1. Open the research Studio

```powershell
.\ASTER_STUDIO.ps1
```

Open **http://127.0.0.1:8765**. The port and restored cream theme are fixed.
Studio exposes experiment results, comparisons, findings, promotion status,
training progress, corpus state, checkpoints, W&B/Hub links, and provider
readiness. It does not store Modal, GCP, W&B, or Hugging Face secrets.

## 2. Understand the selected incumbent

The current model is `configs/model/aster_k3_latentmoe_1p45b_a568m.yaml`:

- 1,448,120,880 total / 568,155,376 active parameters
- 32 layers, hidden width 1,280
- 24 FLA KDA layers + 8 MLA-style latent-attention layers
- 16 routed experts, top-2, plus one shared expert
- packed CUTLASS MoE, BF16 compute, blockwise-INT8 Muon state
- no CPU or NVMe parameter/optimizer/activation offload

It is the incumbent worth serious resources, but Stage 1 remains mechanically
blocked until every required gate passes. Do not bypass a failed gate by editing
YAML.

## 3. Current critical path

1. Complete and import the two-seed 868M-vs-1.448B equal-wall/token/FLOP gate
   using the production Muon memory policy.
2. Materialize, clean, globally deduplicate, benchmark-decontaminate, audit, hash,
   and seal the final corpus; then train and seal its tokenizer.
3. Resolve the code tranche deliberately: obtain the gated NVIDIA data, select an
   audited replacement, or explicitly freeze a no-code mixture.
4. Run only the bounded high-value challengers: GDN2/MLA with the same MoE body,
   and a strong final-scale dense deployment control.
5. Run a multi-hour selected-recipe checkpoint/resume/W&B/Hub/thermal canary.
6. Qualify each cloud GPU/backend/profile with a preflight autotune and explicit
   spend approval before any paid dispatch.

## 4. Data reality

The local raw pool contains about **87.032B tokens**: 84.032B from the primary
frontier corpus plus roughly 3B Nemotron Math tokens. Raw availability is not a
training seal. The final protected run requires committed cleaning reports,
cross-source deduplication, decontamination, disjoint validation, a large manual
audit sample, PII clearance, exact artifact hashes, deliberate source weights,
and sufficient unique clean tokens to keep 100B replay below 2x.

Do not apply fill-in-the-middle augmentation to math. The corpus builder now
requires explicit source identities and applies FIM only to declared code sources.

## 5. Environment and source truth

The validated laptop environment is Python 3.12.3, PyTorch 2.13.0, CUDA runtime
13.0, Triton 3.7.1, FLA 0.5.2, and the package lock at
`requirements/validated-wsl-cu130.txt`. Fedora CUDA 13.1 is a separate supported
target that must pass its own capability probes.

Every consequential campaign runs from a clean, source-pinned checkout. The
current working tree may advance while an experiment runs; its results still
belong to the commit recorded in its manifest.

## 6. Check readiness

```bash
python scripts/check_promotion_gates.py --phase stage1
python scripts/download_status.py --json
python scripts/frontier_vnext_capabilities.py --json runs/setup/capabilities.json
```

Final training uses public Hub checkpoints and automatic newest-compatible resume.
Never start a fresh run before checking whether a later complete Hub checkpoint
exists.

No Modal or GCP spending is implied by these commands. Provider dispatch requires
an explicit profile, spend ceiling, clean contracts, qualified image/backend, and
user confirmation.
