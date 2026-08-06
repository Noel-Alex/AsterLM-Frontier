# AsterLM reasoning-model plan

AsterLM is trained as a **unified thinking/direct model**, not merely a chat model that is prompted to “think step by step.” The reasoning behavior is created by a dedicated post-training curriculum and an explicit tokenizer contract:

- `<|thinking|>` selects deliberate reasoning.
- `<|direct|>` selects a fast direct response.
- `<think> ... </think>` separates private scratch work from the declared result.
- `<answer> ... </answer>` identifies the only answer consumed by deterministic verifiers.

The same checkpoint can therefore trade inference latency for reasoning depth. `scripts/reasoning_chat.py` supports a hard thinking-token budget and closes the thinking section when the budget is exhausted.

## Why the pipeline is staged

Pure RL from a small base model is unnecessarily fragile. It frequently begins with poor formatting, repetitive chains, weak instruction following, and too few correct samples for a group-relative reward to provide useful gradients. The default pipeline follows the more reliable pattern demonstrated by modern reasoning systems:

1. **Knowledge and language pretraining.** Build the model's world knowledge, code familiarity, mathematical language, and long-context behavior.
2. **Broad instruction tuning.** Teach ordinary dialogue and direct responses.
3. **Cold-start reasoning SFT.** Teach readable verified reasoning traces, answer delimiters, and think/direct control tokens.
4. **RL with verifiable rewards (RLVR).** Let the current policy explore multiple solutions and reinforce samples that deterministic verifiers accept.
5. **Optional low-learning-rate mode-fusion refresh.** Restore direct-mode breadth only when direct-mode evaluation measurably regresses; this stage is not applied automatically.
6. **Optional preference alignment.** DPO remains available after reasoning evaluation, but is not assumed to improve mathematical or coding correctness.

## Laptop-specific RL architecture

Multi-GPU RL frameworks often keep an actor, reference, critic, reward model, and rollout server resident at once. That is unsuitable for 12 GiB VRAM. AsterLM instead uses a process-isolated cycle:

```text
current policy checkpoint
        |
        v
GPU process 1: generate grouped rollouts + exact old-policy token log-probabilities
        |
        v
CPU process: deterministic math/code/choice verifiers + rewards + advantages
        |
        v
GPU process 2 (optional): frozen-reference token log-probabilities
        |
        v
GPU process 3: one policy update, save complete checkpoint, commit cycle marker
```

Only one complete model needs GPU memory at any moment. The design is slower, but it preserves on-policy likelihood ratios and allows a frozen-reference KL term without co-resident models.

## Default algorithm: GSPO

`configs/reasoning/rlvr_laptop_gspo.yaml` is the default for the MoE model. It uses the geometric-mean sequence likelihood ratio and sequence-level asymmetric clipping. This is a better starting hypothesis than token-level GRPO for the routed model because a sequence-level objective is less sensitive to token-level routing differences and was proposed specifically with MoE RL stability in mind.

Implemented alternatives:

- **DAPO:** token-level objective, asymmetric clipping, dynamic sampling, and soft overlong-response shaping.
- **GRPO:** group-standardized advantages and per-response token averaging.
- **Dr.GRPO:** removes reward-standard-deviation division and uses global token normalization.
- **REINFORCE baseline:** a minimal unclipped group-baseline control.

The files differ only in configuration, allowing matched experiments on identical rollouts.

## Reward construction

The default reward is deliberately sparse and auditable:

```text
reward = correctness
       + 0.10 * valid answer format
       + 0.05 * completed reasoning section
       - 0.25 * missing final answer
       - 0.10 * repeated 8-gram fraction
       - 0.25 * soft overlong penalty
```

Correctness dominates. Formatting rewards are too small to compensate for a wrong answer.

### Math

The verifier attempts, in order:

1. normalized exact match;
2. tolerant decimal equality;
3. `math-verify` parsing/equivalence;
4. symbolic equivalence through SymPy.

Only the declared final answer after the thinking section is verified. Numbers appearing in intermediate reasoning are never treated as the answer.

### Code

Python answers are extracted from the final answer or final code fence and executed against official tests. The default refuses code execution unless Bubblewrap is installed. The sandbox:

- creates new user/process/network/mount namespaces;
- provides read-only system directories;
- supplies a private temporary work directory;
- disables network access;
- imposes CPU, address-space, process-count, and output-size limits;
- supports stdin/stdout and common function-call tests.

Install on Fedora:

```bash
sudo dnf install bubblewrap
```

Do not enable `allow_unsafe_code_verifier` for downloaded model generations.

## Anti-reward-hacking rules

- The final answer is parsed only after `</think>` or from the explicit `<answer>` section.
- Correctness is deterministic; no trainable reward model is used for math/code RL.
- Missing answers receive a penalty even when the reasoning contains the correct value.
- Repeated reasoning and length-limit exploitation are penalized.
- Zero-variance groups are removed by dynamic sampling.
- Rollouts retain exact token IDs and old-policy log-probabilities; text is not re-tokenized during update.
- Policy/reference scoring gathers target-token log-probabilities in bounded vocabulary chunks instead of retaining full-sequence logits.
- Frozen-reference log-probabilities are precomputed in a separate process when KL regularization is enabled.
- Every cycle has an atomic `update_complete.json` marker, preventing the same on-policy batch from being applied twice after a crash.
- Code execution is isolated and disabled when the sandbox is unavailable.

## Data

The dedicated profile downloads:

| Dataset | Role | Approximate compressed download |
|---|---|---:|
| `open-r1/Mixture-of-Thoughts` (`all`) | 349k verified math/code/science reasoning traces for cold-start SFT | 3.08 GB |
| `open-r1/DAPO-Math-17k-Processed` | 17.4k deduplicated math RL prompts and exact answers | 6 MB |
| `open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled` | 15.1k decontaminated Python RL tasks with executable tests | 146 MB |

The small verifier sets download first; the large trace corpus downloads last. Each source has its own Hugging Face cursor, checksummed local shards, retry log, and atomic state file.

Prepare the local training files:

```bash
python scripts/prepare_reasoning_data.py \
  data/reasoning-frontier \
  --sft-output data/reasoning/reasoning_sft.jsonl \
  --rl-output data/reasoning/rlvr_prompts.jsonl \
  --direct-fraction 0.10

python scripts/prepare_direct_mode_data.py \
  data/posttrain-frontier/smol_smoltalk \
  data/posttrain-frontier/smoltalk2_magpie \
  data/posttrain-frontier/smoltalk2_multilingual \
  data/posttrain-frontier/smoltalk2_science \
  --output data/reasoning/direct_mode_sft.jsonl \
  --max-records 150000
```

RL-only DAPO/code answer rows are not mislabeled as reasoning traces. Cold-start SFT examples are admitted only when an actual assistant reasoning trace is present. A deterministic 10% subset also becomes direct-mode examples, and broader direct-mode instruction data is mixed separately.

## Training commands

Install the additional verifier dependencies:

```bash
bash scripts/setup_linux.sh \
  --skip-torch \
  --with-apollo \
  --with-torchao \
  --with-tracking \
  --with-reasoning
sudo dnf install -y bubblewrap
```

Download only reasoning data:

```bash
python scripts/download_data.py \
  --profile reasoning \
  --validate-first \
  --network-mode low \
  --max-retries 0
```

Or rerun `--profile all`; existing pretraining/post-training checkpoints are retained and the reasoning sources are added.

One-command post-training after a selected base checkpoint exists:

```bash
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k
```

A short validation run should be done first:

```bash
python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --rl-stop-after 2
```

To compare algorithms, copy the SFT checkpoint and invoke `run_reasoning_rl.py` with each reasoning config. Do not compare algorithms after they have already diverged from different policy checkpoints.

## Evaluation gates

Before increasing RL iterations, require:

- rising held-out math pass@1 and pass@k;
- rising code test-pass rate;
- non-decreasing broad instruction quality;
- stable direct-mode response quality;
- controlled response length rather than indiscriminate chain growth;
- non-collapsed reward variance;
- healthy MoE expert utilization and router entropy;
- bounded approximate KL and clipping fraction;
- no sharp increase in repetition or malformed answers.

Use:

```bash
python scripts/evaluate_reasoning.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --checkpoint runs/aster-reasoning-rlvr \
  --reasoning configs/reasoning/rlvr_laptop_gspo.yaml \
  --data data/reasoning/eval_prompts.jsonl \
  --samples 8
```

Interactive budgeted reasoning:

```bash
python scripts/reasoning_chat.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --checkpoint runs/aster-reasoning-rlvr \
  --mode think \
  --thinking-budget 2048 \
  --max-new-tokens 4096
```

## Research references

- DeepSeek-R1, arXiv:2501.12948
- Kimi k1.5, arXiv:2501.12599
- DAPO, arXiv:2503.14476
- Qwen3 Technical Report, arXiv:2505.09388
- Group Sequence Policy Optimization, arXiv:2507.18071

The repository implements laptop-scale versions of the relevant algorithmic ideas. It does not claim to reproduce the scale, throughput, or benchmark results of those frontier training runs.
