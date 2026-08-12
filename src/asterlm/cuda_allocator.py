from __future__ import annotations

from collections.abc import Mapping

DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def cuda_allocator_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return a copied environment with one explicit, consistent PyTorch allocator policy."""

    environment = dict(source)
    modern = environment.get("PYTORCH_ALLOC_CONF")
    legacy = environment.get("PYTORCH_CUDA_ALLOC_CONF")
    if modern and legacy and modern != legacy:
        raise RuntimeError(
            "PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF disagree; "
            "refusing an ambiguous CUDA allocator launch"
        )
    selected = modern or legacy or DEFAULT_CUDA_ALLOC_CONF
    # PYTORCH_ALLOC_CONF is the current name; CUDA_ALLOC_CONF remains an alias
    # in supported PyTorch releases and keeps older pinned environments explicit.
    environment["PYTORCH_ALLOC_CONF"] = selected
    environment["PYTORCH_CUDA_ALLOC_CONF"] = selected
    return environment
