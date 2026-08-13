from __future__ import annotations

import torch

from asterlm.config import AsterConfig
from asterlm.model import AsterLM
from asterlm.training.parameter_policy import apply_parameter_training_policy
from asterlm.training.telemetry import assert_required_gradient_coverage


def _model(*, gradient_checkpointing: bool = False) -> AsterLM:
    return AsterLM(
        AsterConfig(
            vocab_size=128,
            d_model=32,
            n_layers=3,
            n_heads=4,
            head_dim=8,
            ffn_hidden=64,
            ffn_type="latent_moe",
            moe_first_dense_layers=1,
            moe_num_experts=4,
            moe_top_k=2,
            moe_shared_experts=1,
            moe_expert_hidden=32,
            latent_moe_dim=16,
            max_seq_len=32,
            kda_ratio=0,
            latent_rank=8,
            rope_dim=8,
            attention_window=32,
            sink_tokens=0,
            mtp_depth=0,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_segment_size=2,
        )
    )


def test_context_extension_freezes_knowledge_bank_but_not_attention_or_router() -> None:
    model = _model()
    summary = apply_parameter_training_policy(model, "context_extension")
    assert summary["frozen_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["frozen_tensor_count"] == len(summary["frozen_names"])
    for name, parameter in model.named_parameters():
        expected = not (
            (".ffn." in name and ".ffn.router." not in name)
            or name.startswith(("token_embedding.", "lm_head.", "embedding_in_proj.", "embedding_out_proj."))
        )
        assert parameter.requires_grad is expected
    assert not model.token_embedding.weight.requires_grad
    assert any(
        parameter.requires_grad for name, parameter in model.named_parameters() if ".mixer." in name
    )
    assert any(
        parameter.requires_grad for name, parameter in model.named_parameters() if ".ffn.router." in name
    )


def test_all_policy_restores_every_parameter() -> None:
    model = _model()
    apply_parameter_training_policy(model, "context_extension")
    summary = apply_parameter_training_policy(model, "all")
    assert summary["frozen_parameters"] == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_context_extension_preserves_gradients_through_segment_checkpoints() -> None:
    torch.manual_seed(19)
    model = _model(gradient_checkpointing=True).train()
    apply_parameter_training_policy(model, "context_extension")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16))
    labels = torch.randint(0, model.config.vocab_size, (1, 16))
    output = model(input_ids, labels=labels, return_logits=False)
    assert output.loss is not None
    output.loss.backward()
    coverage = assert_required_gradient_coverage(model)
    assert coverage["missing_mixer_blocks"] == []
    assert coverage["embedding_head_tensor_count"] == 0
