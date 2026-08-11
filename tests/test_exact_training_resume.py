from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from asterlm import AsterConfig, AsterLM, DataConfig, TrainConfig
from asterlm.config import SourceConfig
from asterlm.data.tokenizer import SPECIAL_TOKENS
from asterlm.training.checkpoint import load_data_state, load_model_weights, resolve_checkpoint
from asterlm.training.engine import Trainer


def _tokenizer(path: Path) -> int:
    vocabulary = {token: index for index, token in enumerate([*SPECIAL_TOKENS, "[UNK]"])}
    for word in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
        vocabulary[word] = len(vocabulary)
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))
    return len(vocabulary)


def _model(vocab_size: int) -> AsterConfig:
    return AsterConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        head_dim=8,
        ffn_hidden=48,
        max_seq_len=8,
        kda_ratio=0,
        latent_rank=4,
        rope_dim=4,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )


def _train(output: Path, tokenizer: Path, max_steps: int, resume: Path | None = None) -> TrainConfig:
    return TrainConfig(
        output_dir=str(output),
        seed=71,
        deterministic_named_initialization=True,
        device="cpu",
        dtype="float32",
        execution_backend="aster_local",
        compile=False,
        sequence_length=8,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        optimizer="adamw",
        warmup_steps=0,
        schedule_type="constant",
        log_interval=100,
        eval_interval=100,
        save_interval=100,
        keep_last_checkpoints=3,
        tokenizer_path=str(tokenizer),
        jsonl_metrics=False,
        save_diagnostic_bundle=False,
        resume=str(resume) if resume else None,
    )


def _data(path: Path) -> DataConfig:
    return DataConfig(
        sources=[SourceConfig(path=str(path), weight=1.0, fim_rate=0.5)],
        seed=19,
        shuffle_buffer=1,
        min_chars=1,
        quality_filters=False,
    )


def _state(checkpoint: Path, config: AsterConfig) -> dict[str, torch.Tensor]:
    model = AsterLM(config)
    load_model_weights(model, checkpoint)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_uninterrupted_and_reconstructed_training_are_exact(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_size = _tokenizer(tokenizer_path)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"text": "alpha beta gamma delta epsilon zeta " * 8}) + "\n"
            for _ in range(24)
        ),
        encoding="utf-8",
    )
    model_config = _model(vocab_size)
    data_config = _data(corpus)

    continuous_dir = tmp_path / "continuous"
    Trainer(model_config, _train(continuous_dir, tokenizer_path, 2), data_config).train()
    continuous_checkpoint = resolve_checkpoint(continuous_dir)

    resumed_dir = tmp_path / "resumed"
    Trainer(model_config, _train(resumed_dir, tokenizer_path, 1), data_config).train()
    first_checkpoint = resolve_checkpoint(resumed_dir)
    assert load_data_state(first_checkpoint) is not None
    Trainer(
        model_config,
        _train(resumed_dir, tokenizer_path, 2, resume=first_checkpoint),
        data_config,
    ).train()
    resumed_checkpoint = resolve_checkpoint(resumed_dir)

    continuous_state = _state(continuous_checkpoint, model_config)
    resumed_state = _state(resumed_checkpoint, model_config)
    assert continuous_state.keys() == resumed_state.keys()
    for name in continuous_state:
        assert torch.equal(continuous_state[name], resumed_state[name]), name

    continuous_data = load_data_state(continuous_checkpoint)
    resumed_data = load_data_state(resumed_checkpoint)
    assert continuous_data is not None and resumed_data is not None
    continuous_hash = hashlib.sha256(pickle.dumps(continuous_data)).hexdigest()
    resumed_hash = hashlib.sha256(pickle.dumps(resumed_data)).hexdigest()
    assert continuous_hash == resumed_hash
