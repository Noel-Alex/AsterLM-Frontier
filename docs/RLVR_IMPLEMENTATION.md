# RLVR implementation details

## Stored rollout contract

Every sample contains:

- source prompt and verifier metadata;
- `prompt_ids` exactly as supplied to the policy;
- `completion_ids` exactly as sampled;
- one old-policy log-probability per completion token;
- decoded completion for audit and verifier use;
- deterministic rollout seed and finish reason;
- group ID and sample ID;
- reward breakdown, group standard deviation, and advantage;
- optional frozen-reference log-probability per completion token.

The update reconstructs the causal sequence without re-tokenizing text. If the prompt has `P` tokens and completion has `C` tokens, the completion loss mask begins at target index `P-1` and covers exactly `C` targets.

## Advantage estimators

For a group of rewards `r_i`:

- GSPO, DAPO, and GRPO default to `(r_i - mean(r)) / (std(r) + eps)`.
- Dr.GRPO uses only `r_i - mean(r)`.
- Constant-reward groups have zero useful policy gradient and are removed when dynamic sampling is enabled.

## GSPO

For selected completion-token log-probabilities, AsterLM computes the mean log importance ratio over each sequence and exponentiates it. The sequence ratio is asymmetrically clipped to `[1 - clip_low, 1 + clip_high]`; the PPO-style minimum surrogate is then applied once per sequence.

This is the default for the MoE architecture. Router auxiliary and z-loss regularizers are included in the update objective, and bias-based expert balancing is updated after each optimizer step.

## DAPO

DAPO uses token-level importance ratios and asymmetric clipping. The token objective is globally normalized over valid completion tokens rather than independently averaging every response. Dynamic sampling and the soft overlong reward are enabled by the default config.

## GRPO and Dr.GRPO

GRPO independently averages each response's clipped token objective, giving every response equal weight regardless of length. Dr.GRPO removes reward-standard-deviation normalization and uses global token normalization.

## KL regularization

When `kl_beta > 0`, a frozen checkpoint is loaded in a separate process and writes reference log-probabilities into the rollout file. The updater uses the non-negative `exp(x) - x - 1` estimator with `x = log(pi_ref / pi)`. No second model is resident during policy backpropagation.

## VRAM-bounded policy scoring

The policy updater does not retain a full `[batch, sequence, vocabulary]` logits tensor. It requests normalized hidden states from AsterLM, projects them through the output head in `lm_loss_chunk_size` sequence chunks, gathers only the sampled target-token log-probabilities, and checkpoints each chunk during training. The objective is mathematically identical to gathering from a full log-softmax, while peak output-head memory scales with the configured chunk length rather than the complete reasoning trajectory. Frozen-reference scoring uses the same bounded projection without gradient checkpointing.

## Gradient accumulation

Filtered groups may leave a short final accumulation window. Each microbatch is initially divided by the configured accumulation count; before a partial final optimizer step, gradients are rescaled so the effective objective remains the average of the actual microbatches rather than an artificially weakened update.

## Checkpoint and crash semantics

- Rollouts and scored rollouts are immutable cycle artifacts.
- The policy update writes a normal AsterLM checkpoint containing model, optimizer, RNG, step, and token state.
- The orchestrator then atomically writes `cycles/iteration-XXXXXX/update_complete.json`.
- `pipeline_state.json` advances only after the marker exists.
- A restart that finds the marker restores its checkpoint and skips the already-applied cycle.
- A restart before the marker reuses rollout/reference artifacts but performs the missing update.

## Deliberate limitations

- The implementation is single-GPU and sequential, not a high-throughput Ray/vLLM trainer.
- There is no learned critic or process reward model by default.
- General open-ended correctness cannot be safely optimized with deterministic RLVR; current rewards focus on math, multiple choice, and executable Python.
- Full-vocabulary entropy is not retained because it would materially increase activation memory. A selected-token surprise proxy is logged and disabled as a bonus by default.
- The Python KDA/attention implementations remain quality references; sustained rollout speed depends on working FLA and fused decode kernels on the laptop.
