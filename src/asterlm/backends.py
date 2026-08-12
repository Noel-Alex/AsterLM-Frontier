from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Execution policy for one hardware family.

    Backend specs describe eligible implementations. They never alter the model's
    mathematical semantics; each concrete kernel still needs its own parity gate.
    """

    backend_id: str
    device_type: str
    compute_capabilities: tuple[tuple[int, int], ...]
    hardware_family: str
    precision_candidates: tuple[str, ...]
    attention_candidates: tuple[str, ...]
    moe_candidates: tuple[str, ...]
    notes: str

    def supports(self, device_type: str, capability: tuple[int, int] | None) -> bool:
        if self.device_type != device_type:
            return False
        return not self.compute_capabilities or capability in self.compute_capabilities

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}

    def register(self, spec: BackendSpec) -> None:
        if spec.backend_id in self._specs:
            raise ValueError(f"Backend {spec.backend_id!r} is already registered")
        self._specs[spec.backend_id] = spec

    def get(self, backend_id: str) -> BackendSpec:
        try:
            return self._specs[backend_id]
        except KeyError as exc:
            raise KeyError(f"Unknown execution backend {backend_id!r}") from exc

    def resolve(
        self, device_type: str, capability: tuple[int, int] | None = None
    ) -> BackendSpec:
        exact = [
            spec
            for spec in self._specs.values()
            if spec.compute_capabilities and spec.supports(device_type, capability)
        ]
        if len(exact) > 1:
            identifiers = ", ".join(spec.backend_id for spec in exact)
            raise RuntimeError(f"Ambiguous execution backends: {identifiers}")
        if exact:
            return exact[0]
        fallbacks = [
            spec
            for spec in self._specs.values()
            if not spec.compute_capabilities and spec.supports(device_type, capability)
        ]
        if len(fallbacks) != 1:
            raise RuntimeError(
                f"Expected one safe fallback for device_type={device_type!r}; "
                f"found {len(fallbacks)}"
            )
        return fallbacks[0]

    def manifests(self) -> list[dict[str, Any]]:
        return [self._specs[key].manifest() for key in sorted(self._specs)]


def _default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendSpec(
            backend_id="cuda-ada-sm89",
            device_type="cuda",
            compute_capabilities=((8, 9),),
            hardware_family="Ada consumer/laptop",
            precision_candidates=("bf16", "fp8"),
            attention_candidates=("torch_sdpa", "fla_triton", "custom_triton"),
            moe_candidates=(
                "torch_reference",
                "transformer_engine_grouped",
                "cutlass_grouped",
                "torch_grouped",
                "liger_experts",
                "custom_triton",
            ),
            notes="Primary laptop backend; every optional path requires sustained full-model evidence.",
        )
    )
    registry.register(
        BackendSpec(
            backend_id="cuda-hopper-sm90",
            device_type="cuda",
            compute_capabilities=((9, 0),),
            hardware_family="Hopper data-center",
            precision_candidates=("bf16", "fp8"),
            attention_candidates=("torch_sdpa", "flash_attention", "flash_mla", "tile_kernels"),
            moe_candidates=(
                "transformer_engine_grouped",
                "cutlass_grouped",
                "torchao_fp8_grouped",
                "liger_experts",
                "deep_gemm",
                "tile_kernels",
            ),
            notes="Modal candidate; exact GPU type must be pinned for cost comparisons.",
        )
    )
    registry.register(
        BackendSpec(
            backend_id="cuda-blackwell-sm100",
            device_type="cuda",
            compute_capabilities=((10, 0),),
            hardware_family="Blackwell data-center",
            precision_candidates=("bf16", "fp8", "nvfp4"),
            attention_candidates=("torch_sdpa", "flash_mla", "tile_kernels"),
            moe_candidates=(
                "transformer_engine_grouped",
                "torchao_fp8_grouped",
                "liger_experts",
                "deep_gemm",
                "tile_kernels",
            ),
            notes="Distinct from RTX Blackwell; FP4 is quality-gated and never changes canonical weights.",
        )
    )
    registry.register(
        BackendSpec(
            backend_id="cuda-blackwell-rtx-sm120",
            device_type="cuda",
            compute_capabilities=((12, 0),),
            hardware_family="Blackwell RTX",
            precision_candidates=("bf16", "fp8", "nvfp4"),
            attention_candidates=("torch_sdpa", "custom_triton"),
            moe_candidates=(
                "torch_reference",
                "transformer_engine_grouped",
                "liger_experts",
                "custom_triton",
            ),
            notes="Do not assume SM100 kernels work; DeepGEMM V4 gaps have been observed on SM120.",
        )
    )
    registry.register(
        BackendSpec(
            backend_id="cuda-generic",
            device_type="cuda",
            compute_capabilities=(),
            hardware_family="Unknown CUDA",
            precision_candidates=("float32", "bf16", "float16"),
            attention_candidates=("torch_sdpa",),
            moe_candidates=("torch_reference",),
            notes="Safe fallback with no architecture-specific kernel claim.",
        )
    )
    registry.register(
        BackendSpec(
            backend_id="cpu-generic",
            device_type="cpu",
            compute_capabilities=(),
            hardware_family="CPU",
            precision_candidates=("float32",),
            attention_candidates=("torch_sdpa",),
            moe_candidates=("torch_reference",),
            notes="Correctness and portability backend.",
        )
    )
    return registry


EXECUTION_BACKENDS = _default_registry()


def execution_backend_manifest(
    device_type: str, capability: tuple[int, int] | None = None
) -> dict[str, Any]:
    return EXECUTION_BACKENDS.resolve(device_type, capability).manifest()
