# Start here

The repository has been audited for resumable data acquisition, clean local training data, pretraining, ordinary SFT/DPO, and reasoning SFT/RLVR.

## Your current pilot download

The old log shows that FineWeb-Edu, Cosmopedia-v2, FineMath-4+, and DCLM reached approximately **500M tokens in total** before the benchmark stage failed. The fatal `PyGILState_Release` messages occurred after committed source completion; the updated materializers exit cleanly after flushing state. The old MMLU job failed because 15,000 was an impossible cap for the selected split; complete benchmark splits now use `target_records: null`.

Keep the existing local `data/` directory. Verify it, then rerun the pilot command to let the new code recognize completed sources and resume anything genuinely incomplete:

```bash
source .venv/bin/activate
python scripts/download_status.py
python scripts/verify_data_shards.py data/corpus-frontier-16b --only-last
python scripts/download_data.py \
  --profile pilot \
  --validate-first \
  --require-auth \
  --network-mode low \
  --max-retries 0
```

Then download benchmarks, clean the pilot, retrain the tokenizer with the newly added reasoning tokens, and run the training preflight:

```bash
python scripts/download_data.py --profile benchmarks --validate-first --require-auth --network-mode low --max-retries 0

python scripts/prepare_frontier_data.py \
  --raw-corpus data/corpus-frontier-16b \
  --benchmarks data/decontamination-benchmarks \
  --output data/clean-frontier \
  --skip-code

python scripts/train_tokenizer.py \
  --data configs/data/pretrain_frontier_clean.yaml \
  --output artifacts/tokenizer.json \
  --vocab-size 32768

python scripts/training_preflight.py \
  --model configs/model/aster_moe_frontier_893m_a484m.yaml \
  --train configs/train/frontier_stage1_8k.yaml \
  --data configs/data/pretrain_frontier_clean.yaml \
  --check-first-record
```

The complete command set for small-model training, staged frontier pretraining, LoQT, the 1.5B target, SFT, DPO, GSPO/DAPO/GRPO/Dr.GRPO reasoning training, resume, evaluation, inference, and export is in [docs/TRAINING_RUNBOOK.md](docs/TRAINING_RUNBOOK.md).
