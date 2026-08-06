from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class RLSequence:
    input_ids: torch.Tensor
    targets: torch.Tensor
    completion_mask: torch.Tensor
    old_token_logps: torch.Tensor
    reference_token_logps: torch.Tensor | None
    advantage: float
    reward: float
    group_id: int


def build_rl_sequence(record: dict[str, Any]) -> RLSequence:
    prompt_ids = [int(x) for x in record["prompt_ids"]]
    completion_ids = [int(x) for x in record["completion_ids"]]
    old_logps = [float(x) for x in record["old_token_logps"]]
    if not completion_ids or len(completion_ids) != len(old_logps):
        raise ValueError("completion_ids and old_token_logps must be non-empty and aligned")
    full = prompt_ids + completion_ids
    input_ids = torch.tensor(full[:-1], dtype=torch.long)
    targets = torch.tensor(full[1:], dtype=torch.long)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    start = max(0, len(prompt_ids) - 1)
    mask[start : start + len(completion_ids)] = True
    old = torch.zeros_like(targets, dtype=torch.float32)
    old[start : start + len(old_logps)] = torch.tensor(old_logps, dtype=torch.float32)
    ref_values = record.get("reference_token_logps")
    reference = None
    if ref_values is not None:
        if len(ref_values) != len(completion_ids):
            raise ValueError("reference_token_logps must align with completion_ids")
        reference = torch.zeros_like(targets, dtype=torch.float32)
        reference[start : start + len(ref_values)] = torch.tensor(ref_values, dtype=torch.float32)
    return RLSequence(
        input_ids=input_ids,
        targets=targets,
        completion_mask=mask,
        old_token_logps=old,
        reference_token_logps=reference,
        advantage=float(record["advantage"]),
        reward=float(record["reward"]),
        group_id=int(record["group_id"]),
    )


def pad_rl_batch(
    sequences: list[RLSequence],
    pad_id: int,
) -> dict[str, torch.Tensor | None]:
    if not sequences:
        raise ValueError("Cannot pad an empty RL batch")
    length = max(sequence.input_ids.numel() for sequence in sequences)
    batch = len(sequences)
    input_ids = torch.full((batch, length), pad_id, dtype=torch.long)
    targets = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    old = torch.zeros((batch, length), dtype=torch.float32)
    has_reference = any(sequence.reference_token_logps is not None for sequence in sequences)
    reference = torch.zeros((batch, length), dtype=torch.float32) if has_reference else None
    for index, sequence in enumerate(sequences):
        n = sequence.input_ids.numel()
        input_ids[index, :n] = sequence.input_ids
        targets[index, :n] = sequence.targets
        mask[index, :n] = sequence.completion_mask
        old[index, :n] = sequence.old_token_logps
        if reference is not None and sequence.reference_token_logps is not None:
            reference[index, :n] = sequence.reference_token_logps
    return {
        "input_ids": input_ids,
        "targets": targets,
        "completion_mask": mask,
        "old_token_logps": old,
        "reference_token_logps": reference,
        "advantages": torch.tensor([sequence.advantage for sequence in sequences], dtype=torch.float32),
        "rewards": torch.tensor([sequence.reward for sequence in sequences], dtype=torch.float32),
        "group_ids": torch.tensor([sequence.group_id for sequence in sequences], dtype=torch.long),
    }
