# Pretraining scale freeze correction — 2026-08-12

The 2026-08-11 document froze the 269.7M-total K3 proxy before completing the
scale campaign required by `ARCHITECTURE_SELECTION_MANDATE_2026-08-10.md`. That
conclusion is revoked. The proxy's KDA3/MLA, Stable LatentMoE, packed CUTLASS,
Muon, exact-resume and telemetry evidence remains valid engineering input, but it
does not prove that 269.7M total parameters is the best use of a 100B-token budget.

Final scale selection is reopened only among the already-declared K3 tiers:

| Candidate | Logical parameters | Active parameters/token |
|---|---:|---:|
| K3 scale 1 | 868.3M | 483.4M |
| K3 scale 2 | 1.448B | 568.2M |
| K3 stretch | 1.954B | 765.9M |

The smallest proxy is a systems/control model, not a launch winner. Promotion
requires full training-state fit on the 12 GiB laptop, optimized native-context
throughput/VRAM, matched-token and matched-wall learning curves, and long-context
validation. Low utilization is repaired as an execution defect, never used as a
shortcut to reject a quality-leading architecture.
