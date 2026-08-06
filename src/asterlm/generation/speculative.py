from __future__ import annotations

from dataclasses import dataclass

import torch

from asterlm.model import AsterLM


@dataclass(slots=True)
class SpeculativeStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    corrections: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.drafted)


@torch.inference_mode()
def generate_mtp_greedy(
    model: AsterLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    min_draft_confidence: float = 0.0,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """Exact greedy self-speculation using the auxiliary MTP heads.

    This is a transparent reference verifier. It performs full-prefix verification so
    it is exact but not guaranteed to be faster than cached decoding. Its purpose is to
    train/evaluate MTP acceptance before integrating a production tree/CUDA-graph backend.
    """

    if input_ids.shape[0] != 1:
        raise ValueError("MTP reference decoding supports batch size 1")
    if model.mtp is None or model.config.mtp_depth <= 0:
        raise ValueError("The model has no MTP heads")
    context = input_ids
    stats = SpeculativeStats()
    produced = 0

    while produced < max_new_tokens:
        proposal_output = model(context, return_mtp=True)
        main_logits = proposal_output.logits[:, -1]
        drafts = [main_logits.argmax(dim=-1)]
        for head in proposal_output.mtp_logits or []:
            head_logits = head[:, -1]
            confidence = float(head_logits.float().softmax(dim=-1).amax().item())
            if confidence < min_draft_confidence:
                break
            drafts.append(head_logits.argmax(dim=-1))
        drafts = drafts[: max_new_tokens - produced]
        draft_tensor = torch.stack(drafts, dim=1)
        combined = torch.cat((context, draft_tensor), dim=1)
        verification = model(combined)
        start = context.shape[1] - 1
        verifier = verification.logits[:, start : start + len(drafts)].argmax(dim=-1)

        accepted_this_round: list[int] = []
        for index, draft in enumerate(drafts):
            stats.drafted += 1
            proposed = int(draft.item())
            verified = int(verifier[0, index].item())
            if proposed == verified:
                accepted_this_round.append(proposed)
                stats.accepted += 1
                if eos_token_id is not None and proposed == eos_token_id:
                    break
            else:
                accepted_this_round.append(verified)
                stats.corrections += 1
                break

        if not accepted_this_round:
            break
        addition = torch.tensor(accepted_this_round, device=context.device, dtype=context.dtype).unsqueeze(0)
        context = torch.cat((context, addition), dim=1)
        produced += len(accepted_this_round)
        stats.rounds += 1
        if eos_token_id is not None and accepted_this_round[-1] == eos_token_id:
            break
    return context, stats
