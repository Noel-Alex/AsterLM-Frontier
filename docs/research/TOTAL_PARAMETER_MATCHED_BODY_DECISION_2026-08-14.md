# Total-parameter-matched body decision — 2026-08-14

## Decision

The production body remains the 1.448B Stable LatentMoE KDA/MLA model. This
decision is governed by **total parameters**, not active parameters per token.
The earlier 565M active-matched dense experiment is retained as a systems
diagnostic and has no selection authority.

| Candidate | Total parameters | Active/token | Difference from MoE |
|---|---:|---:|---:|
| Stable LatentMoE KDA/MLA | 1,448,120,880 | 568,155,376 | baseline |
| Dense KDA/MLA control | 1,445,907,696 | 1,445,907,696 | -2,213,184 (-0.153%) |

The 0.153% total-size mismatch is small enough to be immaterial at this
experimental resolution and is substantially tighter than changing the dense
FFN width by one practical geometry increment. No active-parameter normalization
is used to choose the body.

## Two-seed result at 1,048,576 tokens per candidate

Both candidates used deterministic name-seeded initialization, identical local
data order, BF16 AMP, per-head Muon with GPU-resident blockwise-INT8 state, and
the same 2K training context. The source checkouts were clean and pinned.

| Seed | MoE eval main loss | Dense eval main loss | MoE tok/s | Dense tok/s |
|---:|---:|---:|---:|---:|
| 1337 | 6.643980 | 6.706405 | 2,394.04 | 2,071.06 |
| 2027 | 6.654390 | 6.726665 | 2,650.57 | 2,174.87 |
| Mean/aggregate | 6.649185 | 6.716535 | 2,522.31 | 2,122.96 |

The MoE is better in both seeds at equal tokens and is 18.8% faster in aggregate.
Its mean loss advantage is 0.06735. Because it is already both faster and lower
loss, the dense control cannot recover the decision under an equal-wall budget.
Peak allocation was also effectively tied: 7.731 GiB for MoE versus 7.658 GiB
for dense.

The dense run's higher point-sampled median utilization (96.25% versus 83.25%)
does not imply higher useful work: it produced fewer tokens per second. At the
production 4K geometry, the selected MoE later sustained 3,339.34 tok/s with
89% median and 98% p90 sampled utilization; the short exact fit gate measured
97% median utilization. Utilization is interpreted together with throughput,
quality, memory, and profiler evidence, never as a standalone target.

## Evidence and provenance

- MoE analysis: `runs/architecture-campaign/final-challenger-fit-2seed-trim-20260813/quality-analysis.json`
- Dense analysis: `runs/architecture-campaign/final-total-matched-dense-fit-2seed-20260814/quality-analysis.json`
- MoE source: `db7f5ab0f713a43386addf5304bc63883ddc4c68`, clean tree manifest `05230d3c684b5505cd8087d36222ffc0ba88134d27122c6b0152ecd882c0caa5`
- Dense source: `d42bce6b24e636536e791fba1c9cb8d703ee2739`, clean tree manifest `5a8065cbd970cffefa3f428bfa9a36a015fea8763f5b181d1568c693cb13366c`
- Dense model: `configs/model/aster_dense_kda3_mla_total_matched_control.yaml`
- Selected MoE model: `configs/model/aster_k3_latentmoe_1p45b_a568m.yaml`

The raw run trees remain in the append-only research archive and Studio index.
This compact tracked decision makes the selection recoverable from Git even when
large exploratory artifacts are later pruned locally.
