from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from asterlm.config import AsterConfig
from asterlm.data import AsterTokenizer
from asterlm.model import AsterLM
from asterlm.training.checkpoint import (
    load_model_weights,
    pin_kda_backend_from_checkpoint,
    resolve_checkpoint,
)
from .sampling import apply_repetition_penalty, sample_next


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.02
    repetition_penalty: float = 1.05
    eos_token_id: int | None = None
    seed: int = 1337
    prefill_chunk_size: int = 2048


def load_runtime(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    model_config: str | Path | None = None,
    device: str = "cuda",
    compile_model: bool = False,
    quantization: str = "none",
) -> tuple[AsterLM, AsterTokenizer]:
    checkpoint = resolve_checkpoint(checkpoint)
    if model_config is None:
        candidate = checkpoint / "model_config.yaml" if checkpoint.is_dir() else None
        if candidate is None or not candidate.exists():
            raise ValueError("Pass --model when the checkpoint has no model_config.yaml")
        model_config = candidate
    config = AsterConfig.from_yaml(model_config)
    pin_kda_backend_from_checkpoint(config, checkpoint)
    model = AsterLM(config)
    load_model_weights(model, checkpoint)
    target_device = torch.device(device)
    # Keep CPU inference in fp32. On Ada CUDA GPUs, bf16 cuts weight memory and
    # bandwidth substantially while retaining the numerics needed by this model.
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested, but CUDA is unavailable")
        model = model.to(device=target_device, dtype=torch.bfloat16)
        # KDA decay parameters are intentionally maintained in fp32 by FLA.
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.data = parameter.data.float()
    else:
        model = model.to(target_device)
    model.eval()

    if quantization != "none":
        try:
            from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        except ImportError as exc:
            raise ImportError("Install torchao to use runtime quantization") from exc

        # Preserve mixer projections used directly by absorbed MLA and custom FLA kernels.
        def ffn_only(module: torch.nn.Module, fqn: str) -> bool:
            return isinstance(module, torch.nn.Linear) and (".ffn." in fqn or ".mtp." in fqn)

        if quantization == "int4":
            qconfig = Int4WeightOnlyConfig(
                group_size=32,
                int4_packing_format="tile_packed_to_4d",
                int4_choose_qparams_algorithm="hqq",
            )
        elif quantization == "int8":
            qconfig = Int8WeightOnlyConfig()
        else:
            raise ValueError("quantization must be one of: none, int4, int8")
        quantize_(model, qconfig, filter_fn=ffn_only)

    if compile_model:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    tokenizer = AsterTokenizer(tokenizer_path)
    return model, tokenizer


@torch.inference_mode()
def generate_tokens(
    model: AsterLM,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> Iterator[int]:
    if input_ids.shape[0] != 1:
        raise ValueError("Streaming generation currently supports batch size 1")
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    cache = model.make_cache()
    chunk_size = max(1, config.prefill_chunk_size)
    output = None
    for start in range(0, input_ids.shape[1], chunk_size):
        output = model(input_ids[:, start : start + chunk_size], cache=cache, use_cache=True)
    assert output is not None and output.logits is not None
    logits = output.logits[:, -1]
    history = input_ids[0]

    for _ in range(config.max_new_tokens):
        penalized = apply_repetition_penalty(logits[0], history, config.repetition_penalty)
        token = sample_next(
            penalized.unsqueeze(0),
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            generator=generator,
        )
        value = int(token.item())
        yield value
        if config.eos_token_id is not None and value == config.eos_token_id:
            break
        history = torch.cat((history, token.view(1)))
        output = model(token.view(1, 1), cache=cache, use_cache=True)
        logits = output.logits[:, -1]


@torch.inference_mode()
def generate(model: AsterLM, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    generated = list(generate_tokens(model, input_ids, config))
    if not generated:
        return input_ids
    suffix = torch.tensor(generated, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
    return torch.cat((input_ids, suffix), dim=1)
