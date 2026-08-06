# Ablation and decision plan

## Promotion rule

A mechanism is promoted only when:

- it fits with safety headroom,
- loss is stable,
- it improves held-out quality at equal tokens,
- downstream results agree,
- its inference benefit is measured,
- complexity is worth the gain.

Close results require multiple seeds.

## A. VRAM matrix

Candidates:

- 670M MoE probe
- 661M dense
- 890M MoE BF16
- 890M MoE FP8
- 890M logical LoQT
- 1.50B MoE
- 1.90B stretch

Axes:

- sequence 2K/4K/8K/16K
- APOLLO-Mini
- TorchAO AdamW4bit
- CPU-offloaded AdamW
- activation offload on/off
- BF16 versus FP8 execution

Reject configurations above about 11.25 GiB peak or with step-to-step allocator growth.

## B. Dense versus MoE

Matched data order and token budget:

- dense 661M
- MoE 670M/364M active
- MoE 890M/482M active
- MoE 1.50B/620M active

Track validation loss by web/math/code, expert load, routing entropy, expert token counts and effective tokens per expert.

## C. Sequence mixer

At a smaller matched model:

- all latent attention
- 1:1 KDA/latent
- 3:1 KDA/latent
- 7:1 KDA/latent
- KDA short convolution on/off
- safe gate on/off
- negative eigenvalues on/off

Evaluate both perplexity and long-context retrieval. Lower cache is not worth severe copying loss.

## D. MLA/cache

- latent rank 64/96/128
- q-LoRA off/on/rank
- RoPE channel 16/32/64
- hot cache 256/1K/2K
- BF16/FP8/INT8/INT4/Hadamard INT4 cold cache
- RoPE key quantization off/on

Report cache GiB, decode speed, perplexity and retrieval degradation.

## E. Precision/memory training

- Muon+AdamW BF16 reference
- APOLLO-Mini
- AdamW8bit
- AdamW4bit
- CPU offload
- Transformer Engine FP8
- LoQT rank 32/64/128 and merge interval

Do not compare only throughput. Compare validation loss at equal tokens and wall-clock, plus peak VRAM.

## F. Outlier-safe training

Factorial ablation:

- RMSNorm / SSNorm
- embedding projection off/on
- Muon / alternative optimizer

Measure activation kurtosis, max channel magnitude, quantized perplexity and 4-bit downstream performance. Use `export_osp_merged.py` before inference comparison.

## G. MTP and speculative decoding

- MTP off/depth 1/depth 2/depth 3
- loss weight sweep
- acceptance by domain
- exact verifier versus ordinary decode

Promote as speed optimization only if warmed end-to-end decode improves.

## H. Data

- web-heavy baseline
- DCLM fraction
- Cosmopedia fraction
- FineMath fraction
- code 5/10/15/20%
- language distribution in code
- reasoning continuation length

Track effective epochs and avoid repeatedly cycling a small high-quality subset until it overfits.

## I. Context

- 8K only
- 8K→16K
- 8K→16K→32K
- optional 64K continuation

Tests:

- needle depth 10/50/90%
- repeated-key interference
- many distractors
- multi-hop retrieval
- long code dependencies
- natural-document perplexity

## Result schema

```text
run_id, git_sha, model, logical_params, active_params, trainable_params,
tokens, sequence, optimizer, precision, peak_vram, tok_s,
val_web, val_math, val_code, downstream_json,
needle_8k, needle_32k, needle_64k, cache_dtype,
expert_load_cv, activation_kurtosis, mtp_accept_1, mtp_accept_2
```
