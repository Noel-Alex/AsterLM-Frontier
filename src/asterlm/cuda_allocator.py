from __future__ import annotations

from collections.abc import Mapping

# The RTX 4080 Laptop WSL/CUDA 13.0 validation stack has repeatedly failed
# expandable virtual-memory mappings despite several GiB remaining free. Keep
# the portable default on the native allocator; providers may override it only
# through an explicit, recorded, source-pinned qualification recipe.
# Do not add a split-size cap here. The selected 1.448B model sustained one
# accumulated update with that heuristic, then failed during the next update
# despite several GiB reported free. The unmodified native allocator completed
# beyond the same boundary under the identical source-pinned recipe.
DEFAULT_CUDA_ALLOC_CONF = "backend:native"


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
