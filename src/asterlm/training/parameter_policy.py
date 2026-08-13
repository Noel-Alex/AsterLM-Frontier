from __future__ import annotations

import hashlib
import json
from typing import Any

from torch import nn


def apply_parameter_training_policy(model: nn.Module, policy: str) -> dict[str, Any]:
    """Apply and describe one explicit, checkpoint-compatible training policy."""

    if policy not in {"all", "context_extension"}:
        raise ValueError(f"Unsupported parameter training policy: {policy}")
    frozen_names: list[str] = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        lower = name.lower()
        frozen = policy == "context_extension" and (
            # Context continuation preserves the learned knowledge bank. The
            # frozen embedding activation is explicitly made gradient-bearing in
            # AsterLM.forward so reentrant checkpointing still differentiates all
            # attention/recurrent blocks without allocating embedding optimizer
            # state.
            (".ffn." in lower and ".ffn.router." not in lower)
            or lower.startswith(
                ("token_embedding.", "lm_head.", "embedding_in_proj.", "embedding_out_proj.")
            )
        )
        parameter.requires_grad_(not frozen)
        if frozen:
            frozen_names.append(name)
            frozen_parameters += parameter.numel()
        else:
            trainable_parameters += parameter.numel()
    encoded = json.dumps(frozen_names, separators=(",", ":")).encode("utf-8")
    return {
        "policy": policy,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "frozen_tensor_count": len(frozen_names),
        "frozen_names_sha256": hashlib.sha256(encoded).hexdigest(),
        "frozen_names": frozen_names,
    }
