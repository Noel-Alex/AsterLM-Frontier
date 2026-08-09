# AsterLM vNext2 Research Charter

## Governing rule

Aster is not a MoE project, a dense-model project, a KDA project, or a long-context demo. It is a constrained architecture–systems research program: **choose the model that produces the best measured quality/efficiency frontier on the actual RTX 4080 Laptop GPU.** Every fashionable component is an ablation until it earns its place.

The final billion-class run should be expensive only after small controlled experiments have removed obvious systems mistakes and architecture confounds. A failed idea is useful data when the experiment is controlled and retained.

## What counts as a win

The campaign tracks several objectives separately rather than hiding them in one score:

- main-model validation loss at fixed training tokens;
- main-model validation loss at approximately fixed wall-clock;
- training tokens/s and peak/reserved VRAM;
- active and total parameters;
- long-context feasibility and throughput;
- stability: finite losses/gradients, router balance, QK behavior;
- later, task quality and inference latency/cache bandwidth.

Dense and sparse models must be compared at sensible *active compute*, not by misleading total parameter counts alone. Short screens are followed by LR sweeps and longer confirmations so a single bad learning rate or lucky early trajectory cannot choose the architecture.

## Experimental hygiene

- Each performance/training experiment executes in a fresh child process. Process exit destroys that PyTorch CUDA context and allocator cache.
- Before and after every trial, vNext2 queries `nvidia-smi`, checks compute PIDs, VRAM, utilization and temperature, and waits for an idle baseline. It never kills a user process automatically.
- Performance experiments require AC power when Linux can determine adapter state.
- Important low-level A/Bs use A/B/B/A ordering to reduce warm-up, thermal and run-order bias.
- Results are resumable and failures/OOMs are retained. Transient busy/timeout/contaminated trials are **not** cached as scientific results and are eligible for rerun.
- The runner places every child in its own process group; a timeout terminates descendants as well, preventing an orphaned CUDA process from polluting later trials.
- While a trial runs, the parent periodically records `nvidia-smi` state and flags any unrelated CUDA compute PID that appears mid-run.
- Exact mathematical rewrites (e.g. absorbed MLA) require forward **and gradient** parity tests before speed claims.
- Proxy data/tokenizer artifacts are explicitly research controls, not the final corpus/tokenizer.

## Long-context objective

256K training and 1M inference remain aspirational targets, not marketing labels. A context length counts only when the model can use it meaningfully and the runtime is viable. KV storage fitting in VRAM is necessary but insufficient: indexer cost, memory locality, dequantization, retrieval quality, prefill and decode bandwidth all matter.

## Publication rule

A paper/study is earned only if experiments reveal a reproducible result: e.g. an architecture–systems Pareto improvement, a consumer-GPU-specific systems finding, or an Aster-specific method that survives strong baselines and ablations. We will not claim novelty merely because several recent components were combined.
