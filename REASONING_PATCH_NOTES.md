# AsterLM reasoning-model patch

This patch upgrades the Linux v2 repository into a unified **thinking/direct** model stack. It is safe to extract over an existing checkout: downloaded corpus shards, Hugging Face caches, cursor checkpoints, logs, and trained checkpoints are not included in the patch and are not deleted.

## Apply to an existing Fedora checkout

Stop a running downloader with **Ctrl-C once** so it commits its current cursor, then run:

```bash
cd ~/Documents/AsterLM-Frontier
unzip -o ~/Downloads/AsterLM-Reasoning-Model-Patch.zip -d .
source .venv/bin/activate
pip install -e ".[reasoning]"
sudo dnf install -y bubblewrap
```

Resume the complete low-bandwidth graph with the same command:

```bash
python scripts/download_data.py \
  --profile all \
  --validate-first \
  --network-mode low \
  --max-retries 0
```

Existing source cursors are reused. The new `reasoning` stage adds the small DAPO/code verifier datasets first and the approximately 3.08 GB Mixture-of-Thoughts source last. To download only the new sources, use `--profile reasoning`.

## Important tokenizer note

The reasoning system adds `<|thinking|>`, `<|direct|>`, `<think>`, `</think>`, `<answer>`, and `</answer>`. If a tokenizer was already trained before applying this patch, retrain it before pretraining or reasoning post-training. Existing corpus downloads do not need to be repeated.

## Prepare and train

```bash
python scripts/prepare_reasoning_data.py \
  data/reasoning-frontier \
  --sft-output data/reasoning/reasoning_sft.jsonl \
  --rl-output data/reasoning/rlvr_prompts.jsonl \
  --direct-fraction 0.10

python scripts/run_reasoning_posttrain.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --base-checkpoint runs/aster-frontier-stage3-32k \
  --rl-stop-after 2
```

The two-iteration run is the mandatory initial validation. Inspect reward variance, pass rates, malformed answers, completion length, approximate KL, clipping fraction, and MoE load balance before increasing the iteration count.

## Implemented reasoning components

- Verified long-form cold-start SFT plus direct-mode examples.
- Explicit think/direct control tokens and hard reasoning budgets.
- Exact on-policy grouped rollouts with stored old-policy token log-probabilities.
- GSPO, DAPO, GRPO, Dr.GRPO, and REINFORCE-baseline objectives.
- Separate-process frozen-reference KL scoring.
- Exact/symbolic math, multiple-choice, and sandboxed Python rewards.
- Dynamic removal of zero-variance groups, repetition/format/overlong safeguards.
- Atomic per-cycle completion markers and resumable policy updates.
- Router auxiliary/z regularization and bias-based expert balancing during RL.
- Checkpointed, vocabulary-chunked selected-token scoring to avoid full-sequence logits.
- Optional low-learning-rate direct/reasoning fusion refresh.

Validation: **48 tests passed**, Python compilation passed, all YAML configurations parsed, and the complete downloader graph passed a dry run. CUDA throughput and peak VRAM still need measurement on the RTX 4080 Laptop GPU.
