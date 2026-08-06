# Speculative decoding

## Implemented native path

AsterLM trains two MTP future-token heads. The exact greedy reference:

1. obtains the base next token and MTP proposals,
2. proposes up to two future tokens,
3. verifies with the target model,
4. accepts the longest exact prefix,
5. falls back to the target token on mismatch.

```bash
python scripts/benchmark_speculative.py \
  --checkpoint artifacts/aster-osp-folded \
  --new-tokens 128
```

The tests verify exact greedy-output equality.

## Why this is not automatically fast

The included verifier emphasizes correctness and acceptance measurement. A Python verifier that recomputes too much can be slower than normal decoding even with high acceptance. Real speed requires:

- cache-aware block verification
- fused target verification
- tree/block attention
- stable CUDA shapes
- sufficiently large target-model cost to amortize drafting

## DeepSpec/EAGLE path

The repository includes `scripts/export_speculative_cache.py` and `integrations/deepspec/README.md` to expose:

- target model configuration
- selected hidden features
- tokenizer/special tokens
- MTP head metadata
- checkpoint paths

A production EAGLE-3/DeepSpec experiment requires a separate draft-training job and a compatible serving engine. It is not reduced to a misleading boolean flag.

## Experimental ladder

1. Measure MTP depth-1/depth-2 accuracy during validation.
2. Verify exact greedy equality.
3. Add a cache-aware target verifier.
4. Measure draft, verification and fallback time separately.
5. Integrate EAGLE-3 only if the target is expensive enough.
6. Compare a block-diffusion drafter for code/math workloads only after a conventional drafter baseline.

## Metrics

- proposal rounds
- drafted tokens
- accepted tokens by depth
- mean accepted length
- mismatch position
- draft latency
- target verification latency
- fallback latency
- warmed end-to-end tokens/s
- output equality for greedy mode
- distributional correctness for sampled mode

The current guarantee is exact greedy equivalence. Temperature sampling needs proper speculative acceptance/rejection correction and must not be approximated silently.
