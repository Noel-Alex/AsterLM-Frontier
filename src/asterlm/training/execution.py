from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata, util
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml
from torch import nn

from asterlm.config import AsterConfig, TrainConfig


@dataclass(frozen=True, slots=True)
class ExecutionTopology:
    """Hardware/process topology captured before an execution plan is selected."""

    device_type: str
    world_size: int
    local_world_size: int
    rank: int
    local_rank: int
    cuda_device_count: int
    cuda_capability: tuple[int, int] | None
    cuda_device_name: str | None

    @classmethod
    def detect(cls, device: torch.device) -> ExecutionTopology:
        capability: tuple[int, int] | None = None
        name: str | None = None
        if device.type == "cuda" and torch.cuda.is_available():
            index = device.index if device.index is not None else torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(index)
            name = torch.cuda.get_device_name(index)
        return cls(
            device_type=device.type,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            cuda_capability=capability,
            cuda_device_name=name,
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Resolved, auditable training execution decisions for one run.

    Architecture and execution are deliberately separate: selecting this plan must
    never silently replace KDA, MLA, MoE, MTP, or their mathematical definitions.
    """

    engine: str
    topology: ExecutionTopology
    compile_enabled: bool
    compile_mode: str | None
    compile_dynamic: bool
    precision_backend: str
    activation_offload: bool
    cuda_graphs: bool
    distributed_strategy: str
    decisions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionBackendCapability:
    """Evidence about a runtime candidate; availability is not promotion."""

    backend: str
    module: str
    distribution: str | None
    importable: bool
    installed_version: str | None
    source_repository: str | None
    source_path: str | None
    expected_commit: str | None
    observed_commit: str | None
    source_matches_lock: bool | None
    adapter_implemented: bool
    promoted: bool
    topology_supported: bool
    usable: bool
    blockers: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BACKEND_PACKAGES: dict[str, tuple[str, str | None]] = {
    "aster_local": ("asterlm", "asterlm"),
    "megatron_core": ("megatron.core", "megatron-core"),
    "torchtitan": ("torchtitan", "torchtitan"),
    "deepspeed": ("deepspeed", "deepspeed"),
}


def _find_module(module: str) -> bool:
    try:
        return util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(distribution: str | None) -> str | None:
    if distribution is None:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _source_locks(lock_path: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    path = Path(lock_path)
    if not path.is_file():
        return Path("/root/.cache/asterlm/upstreams"), {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(str(raw.get("download_root", "/root/.cache/asterlm/upstreams")))
    sources = {
        str(item["id"]): dict(item)
        for item in raw.get("sources", [])
        if isinstance(item, dict) and item.get("id")
    }
    return root, sources


def probe_execution_backends(
    device: torch.device,
    *,
    lock_path: str | Path = "configs/research/upstream_sources_2026-08-10.yaml",
    source_root: str | Path | None = None,
) -> dict[str, ExecutionBackendCapability]:
    """Report runtime evidence without importing heavyweight backend packages.

    The report is safe to expose in manifests and the control UI. A cloned checkout
    or installed package is recorded independently from Aster adapter promotion.
    """

    topology = ExecutionTopology.detect(device)
    locked_root, locks = _source_locks(lock_path)
    root = Path(source_root) if source_root is not None else locked_root
    capabilities: dict[str, ExecutionBackendCapability] = {}
    for backend, (module, distribution) in _BACKEND_PACKAGES.items():
        local = backend == "aster_local"
        lock = locks.get(backend)
        repository = str(lock["repository"]) if lock and lock.get("repository") else None
        expected = str(lock["commit"]) if lock and lock.get("commit") else None
        checkout: Path | None = None
        observed: str | None = None
        matches: bool | None = None
        if repository:
            checkout = root / Path(repository.removesuffix(".git")).name
            observed = _git_commit(checkout)
            matches = observed == expected if observed is not None and expected is not None else False

        importable = True if local else _find_module(module)
        version = _distribution_version(distribution)
        if local and version is None:
            version = "editable"
        adapter_implemented = local
        promoted = local
        topology_supported = topology.world_size == 1 if local else False
        blockers: list[str] = []
        if not importable:
            blockers.append("runtime package is not importable")
        if lock and observed is None:
            blockers.append("pinned source checkout is absent or unreadable")
        elif lock and not matches:
            blockers.append("source checkout does not match the pinned commit")
        if not adapter_implemented:
            blockers.append("Aster adapter is not implemented")
        if not promoted:
            blockers.append("numerical, recovery, quality, and throughput gates have not passed")
        if not topology_supported:
            blockers.append("adapter has not been validated for the detected topology")

        usable = importable and adapter_implemented and promoted and topology_supported
        capabilities[backend] = ExecutionBackendCapability(
            backend=backend,
            module=module,
            distribution=distribution,
            importable=importable,
            installed_version=version,
            source_repository=repository,
            source_path=str(checkout) if checkout is not None else None,
            expected_commit=expected,
            observed_commit=observed,
            source_matches_lock=matches,
            adapter_implemented=adapter_implemented,
            promoted=promoted,
            topology_supported=topology_supported,
            usable=usable,
            blockers=tuple(blockers),
        )
    return capabilities


class TrainingExecutionEngine(Protocol):
    """Small boundary that lets Aster compare runtimes without forking model math."""

    plan: ExecutionPlan

    def prepare_model(self, model: nn.Module) -> nn.Module: ...

    def backward(self, loss: torch.Tensor) -> None: ...

    def optimizer_step(self, optimizer: Any) -> None: ...


class AsterLocalExecutionEngine:
    """Lean single-process engine used by the laptop and single-GPU controls."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    def prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.plan.compile_enabled:
            return model
        return torch.compile(
            model,
            mode=self.plan.compile_mode or "default",
            dynamic=self.plan.compile_dynamic,
        )

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def optimizer_step(optimizer: Any) -> None:
        optimizer.step()


def resolve_execution_engine(
    model_config: AsterConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> AsterLocalExecutionEngine:
    """Resolve the executable plan, rejecting any runtime that is only aspirational.

    Megatron Core, TorchTitan, and DeepSpeed are campaign candidates. They are not
    reported as active until their Aster adapters pass numerical, recovery, and
    throughput gates on the exact target hardware.
    """

    topology = ExecutionTopology.detect(device)
    requested = train_config.execution_backend
    if requested not in {"auto", "aster_local"}:
        raise NotImplementedError(
            f"execution_backend={requested!r} has no promoted Aster adapter yet; "
            "run the execution-engine campaign before enabling it for training"
        )
    if topology.world_size != 1:
        raise RuntimeError(
            "Aster's local execution plan is single-process. Select a validated "
            "distributed adapter instead of silently running only one rank."
        )
    if train_config.cuda_graphs:
        raise NotImplementedError(
            "CUDA-graph training is not promoted yet; capture must first pass static-shape, "
            "optimizer, checkpoint, and MoE routing-parity gates"
        )

    compile_enabled = train_config.compile
    decisions: list[str] = ["single-process Aster execution plan"]
    if compile_enabled and model_config.linear_backend == "transformer_engine":
        raise ValueError(
            "Compile and Transformer Engine remain an experimental combination; "
            "select one until the exact model path passes parity and stability gates"
        )
    if compile_enabled:
        decisions.append("static-shape torch.compile model region")
    else:
        decisions.append("eager model region")
    decisions.append(f"precision={train_config.precision_backend}/{train_config.dtype}")
    if train_config.activation_offload:
        decisions.append("saved-tensor CPU activation offload")

    plan = ExecutionPlan(
        engine="aster_local",
        topology=topology,
        compile_enabled=compile_enabled,
        compile_mode=train_config.compile_mode if compile_enabled else None,
        compile_dynamic=False,
        precision_backend=train_config.precision_backend,
        activation_offload=train_config.activation_offload,
        cuda_graphs=False,
        distributed_strategy="none",
        decisions=tuple(decisions),
    )
    return AsterLocalExecutionEngine(plan)
