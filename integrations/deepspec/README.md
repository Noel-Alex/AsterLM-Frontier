# DeepSpec integration

AsterLM trains native multi-token-prediction heads and includes an exact verifier for
acceptance measurement. For production speculative decoding, use DeepSeek's **DeepSpec**
stack rather than pretending the Python reference verifier is fast.

1. Finish base/SFT training and benchmark the built-in MTP heads.
2. Create domain-matched prompt/answer sequences.
3. Export compact hidden-state and top-k target caches:

```bash
python scripts/export_speculative_cache.py \
  --checkpoint runs/aster-frontier-sft \
  --input data/speculative/train.jsonl \
  --output data/speculative/target-cache \
  --sequence-length 4096 --top-k 32
```

4. Clone `deepseek-ai/DeepSpec` into a separate environment. Its released configs assume
large Hugging Face targets and multiple GPUs, so add an AsterLM target adapter around the
export format above rather than attempting to load AsterLM through `AutoModel`.
5. Compare DSpark, DFlash and EAGLE3 on accepted tokens per target call **and** end-to-end
tokens/s. Do not choose by acceptance rate alone.

DeepSpec target caches can become enormous when full logits and many hidden layers are
stored. This exporter keeps one final hidden state and top-k logits; raise `--top-k` only
if draft quality needs it.
