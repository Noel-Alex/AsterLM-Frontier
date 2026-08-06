from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from asterlm.config import TrainConfig
from .muon import Muon


@dataclass
class OptimizerPartition:
    muon_names: list[str]
    adam_decay_names: list[str]
    adam_no_decay_names: list[str]


class HybridOptimizer:
    """Small wrapper that presents Muon and AdamW as one optimizer."""

    def __init__(self, muon: Muon | None, adamw: torch.optim.AdamW | None, partition: OptimizerPartition) -> None:
        self.muon = muon
        self.adamw = adamw
        self.partition = partition
        self._base_lrs = {
            "muon": None if muon is None else [g["lr"] for g in muon.param_groups],
            "adamw": None if adamw is None else [g["lr"] for g in adamw.param_groups],
        }

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def set_lr_multiplier(self, multiplier: float) -> None:
        if self.muon is not None and self._base_lrs["muon"] is not None:
            for group, base in zip(self.muon.param_groups, self._base_lrs["muon"], strict=True):
                group["lr"] = base * multiplier
        if self.adamw is not None and self._base_lrs["adamw"] is not None:
            for group, base in zip(self.adamw.param_groups, self._base_lrs["adamw"], strict=True):
                group["lr"] = base * multiplier

    def state_dict(self) -> dict:
        return {
            "muon": None if self.muon is None else self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
            "partition": self.partition.__dict__,
            "base_lrs": self._base_lrs,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None and state.get("muon") is not None:
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state.get("adamw") is not None:
            self.adamw.load_state_dict(state["adamw"])
        self._base_lrs = state.get("base_lrs", self._base_lrs)


def _unique_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        yield name, param


def build_hybrid_optimizer(model: nn.Module, config: TrainConfig) -> HybridOptimizer:
    muon: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []
    muon_names: list[str] = []
    adam_decay_names: list[str] = []
    adam_no_decay_names: list[str] = []

    for name, param in _unique_named_parameters(model):
        lower = name.lower()
        is_osp_projection = lower.startswith("embedding_in_proj") or lower.startswith("embedding_out_proj")
        is_embedding_or_head = ("embedding" in lower and not is_osp_projection) or lower.startswith("lm_head")
        is_matrix = param.ndim == 2
        is_norm_or_bias = param.ndim < 2 or lower.endswith("bias") or "norm" in lower or "router" in lower
        # Depthwise/short convolutions and special recurrence-time constants are not
        # ordinary hidden-state matrices, so keep them on AdamW rather than Muon.
        is_special_adam = "conv" in lower or "a_log" in lower or "dt_bias" in lower or "router" in lower
        # Muon is reserved for dense hidden matrices. Tied embeddings/output heads use
        # AdamW with no decay, following stability-oriented small-model recipes.
        if is_matrix and not is_embedding_or_head and not is_special_adam:
            muon.append(param)
            muon_names.append(name)
        elif is_norm_or_bias or is_embedding_or_head:
            adam_no_decay.append(param)
            adam_no_decay_names.append(name)
        else:
            adam_decay.append(param)
            adam_decay_names.append(name)

    muon_optim = (
        Muon(
            muon,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
            ns_steps=config.muon_ns_steps,
            nesterov=config.muon_nesterov,
            update_rms=config.muon_update_rms,
        )
        if muon
        else None
    )
    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": config.weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    adam_optim = (
        torch.optim.AdamW(
            adam_groups,
            lr=config.adam_lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            fused=any(param.is_cuda for group in adam_groups for param in group["params"]),
        )
        if adam_groups
        else None
    )
    partition = OptimizerPartition(muon_names, adam_decay_names, adam_no_decay_names)
    return HybridOptimizer(muon_optim, adam_optim, partition)


class SingleOptimizerAdapter:
    """Expose a standard torch optimizer through the trainer's small optimizer protocol."""

    def __init__(self, optimizer: torch.optim.Optimizer, partition: OptimizerPartition) -> None:
        self.optimizer = optimizer
        self.partition = partition
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]

    @property
    def param_groups(self) -> list[dict]:
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.optimizer.step()

    def set_lr_multiplier(self, multiplier: float) -> None:
        for group, base in zip(self.optimizer.param_groups, self._base_lrs, strict=True):
            group["lr"] = base * multiplier

    def state_dict(self) -> dict:
        return {
            "kind": "single",
            "optimizer": self.optimizer.state_dict(),
            "partition": self.partition.__dict__,
            "base_lrs": self._base_lrs,
        }

    def load_state_dict(self, state: dict) -> None:
        payload = state.get("optimizer", state)
        self.optimizer.load_state_dict(payload)
        self._base_lrs = state.get("base_lrs", self._base_lrs)


def build_apollo_optimizer(model: nn.Module, config: TrainConfig) -> SingleOptimizerAdapter:
    """Build APOLLO or APOLLO-Mini without vendoring its non-MIT implementation.

    Install with ``pip install apollo-torch``. Almost all dense 2-D matrices use the
    low-rank APOLLO state. Embeddings, tied output weights, routers, norms, biases,
    convolution kernels, and recurrent constants remain regular Adam-style groups.
    """
    try:
        from apollo_torch import APOLLOAdamW
    except ImportError as exc:
        raise ImportError(
            "APOLLO was requested. Install the optional dependency with "
            "`pip install -e '.[memory]'` or `pip install apollo-torch`."
        ) from exc

    projected_decay: list[nn.Parameter] = []
    regular_decay: list[nn.Parameter] = []
    regular_no_decay: list[nn.Parameter] = []
    projected_names: list[str] = []
    regular_decay_names: list[str] = []
    regular_no_decay_names: list[str] = []

    for name, param in _unique_named_parameters(model):
        lower = name.lower()
        is_osp_projection = lower.startswith("embedding_in_proj") or lower.startswith("embedding_out_proj")
        embedding_or_head = ("embedding" in lower and not is_osp_projection) or lower.startswith("lm_head")
        sensitive = "router" in lower or "norm" in lower or lower.endswith("bias")
        special = "conv" in lower or "a_log" in lower or "dt_bias" in lower
        if param.ndim == 2 and not embedding_or_head and not sensitive and not special:
            projected_decay.append(param)
            projected_names.append(name)
        elif param.ndim < 2 or embedding_or_head or sensitive:
            regular_no_decay.append(param)
            regular_no_decay_names.append(name)
        else:
            regular_decay.append(param)
            regular_decay_names.append(name)

    rank = 1 if config.optimizer == "apollo_mini" else config.apollo_rank
    scale_type = "tensor" if config.optimizer == "apollo_mini" else config.apollo_scale_type
    groups: list[dict] = []
    if projected_decay:
        groups.append(
            {
                "params": projected_decay,
                "weight_decay": config.weight_decay,
                "rank": rank,
                "proj": config.apollo_proj,
                "scale_type": scale_type,
                "scale": config.apollo_scale,
                "update_proj_gap": config.apollo_update_proj_gap,
                "proj_type": config.apollo_proj_type,
            }
        )
    if regular_decay:
        groups.append({"params": regular_decay, "weight_decay": config.weight_decay})
    if regular_no_decay:
        groups.append({"params": regular_no_decay, "weight_decay": 0.0})

    optimizer = APOLLOAdamW(
        groups,
        lr=config.adam_lr,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=0.0,
        scale_front=config.apollo_scale_front,
        disable_nl=config.apollo_disable_norm_limiter,
    )
    partition = OptimizerPartition(projected_names, regular_decay_names, regular_no_decay_names)
    return SingleOptimizerAdapter(optimizer, partition)


def _adamw_parameter_groups(model: nn.Module, config: TrainConfig) -> tuple[list[dict], OptimizerPartition]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, param in _unique_named_parameters(model):
        lower = name.lower()
        no_wd = (
            param.ndim < 2
            or lower.endswith("bias")
            or "norm" in lower
            or "embedding" in lower
            or lower.startswith("lm_head")
            or "router" in lower
            or "a_log" in lower
            or "dt_bias" in lower
        )
        if no_wd:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)
            decay_names.append(name)
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups, OptimizerPartition([], decay_names, no_decay_names)


def build_torchao_optimizer(model: nn.Module, config: TrainConfig) -> SingleOptimizerAdapter:
    try:
        from torchao.optim import AdamW4bit, AdamW8bit, CPUOffloadOptimizer
    except ImportError as exc:
        raise ImportError(
            "A torchao optimizer was requested. Install a torch/CUDA-matched torchao wheel "
            "using scripts/setup_linux.sh --with-torchao."
        ) from exc
    groups, partition = _adamw_parameter_groups(model, config)
    kwargs = {
        "lr": config.adam_lr,
        "betas": config.adam_betas,
        "eps": config.adam_eps,
        "weight_decay": 0.0,
    }
    if config.optimizer == "torchao_adamw8bit":
        optimizer = AdamW8bit(groups, **kwargs)
    elif config.optimizer == "torchao_adamw4bit":
        optimizer = AdamW4bit(groups, **kwargs)
    elif config.optimizer == "torchao_cpu_offload_adamw":
        # The wrapped AdamW keeps optimizer state and copied gradients on CPU. This is
        # usually the largest single VRAM reduction, at the cost of PCIe traffic.
        optimizer = CPUOffloadOptimizer(
            groups,
            torch.optim.AdamW,
            **kwargs,
            fused=True,
        )
    else:  # pragma: no cover - caller validates
        raise ValueError(config.optimizer)
    return SingleOptimizerAdapter(optimizer, partition)


def build_optimizer(model: nn.Module, config: TrainConfig) -> HybridOptimizer | SingleOptimizerAdapter:
    if config.optimizer == "muon_adamw":
        return build_hybrid_optimizer(model, config)
    if config.optimizer in {"apollo_mini", "apollo"}:
        return build_apollo_optimizer(model, config)
    return build_torchao_optimizer(model, config)
