#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml

from asterlm.config import AsterConfig
from asterlm.model import AsterLM
from asterlm.training.checkpoint import load_model_weights, pin_kda_backend_from_checkpoint, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold OSP embedding projections into untied embedding/head weights"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Optional model YAML; checkpoint config is preferred")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    config_path = Path(args.model) if args.model else checkpoint / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find model configuration: {config_path}")
    source_config = AsterConfig.from_yaml(config_path)
    pin_kda_backend_from_checkpoint(source_config, checkpoint)
    if not source_config.embedding_projection:
        raise ValueError("The source checkpoint does not enable embedding_projection")

    source = AsterLM(source_config)
    load_model_weights(source, checkpoint)
    source.eval()
    folded_embedding, folded_head = source.folded_embedding_weights()

    export_config = copy.deepcopy(source_config)
    export_config.embedding_projection = False
    # The input/output rotations differ, so folding intentionally produces untied matrices.
    export_config.tie_embeddings = False
    exported = AsterLM(export_config)
    source_state = source.state_dict()
    export_state = exported.state_dict()
    skipped = {
        "token_embedding.weight",
        "lm_head.weight",
        "embedding_in_proj.weight",
        "embedding_out_proj.weight",
    }
    compatible = {
        name: value
        for name, value in source_state.items()
        if name not in skipped and name in export_state and export_state[name].shape == value.shape
    }
    missing, unexpected = exported.load_state_dict(compatible, strict=False)
    allowed_missing = {"token_embedding.weight", "lm_head.weight"}
    bad_missing = set(missing) - allowed_missing
    if bad_missing or unexpected:
        raise RuntimeError(f"Unexpected fold mismatch: missing={sorted(bad_missing)}, unexpected={unexpected}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    with torch.no_grad():
        exported.token_embedding.weight.copy_(folded_embedding.to(dtype=exported.token_embedding.weight.dtype))
        exported.lm_head.weight.copy_(folded_head.to(dtype=exported.lm_head.weight.dtype))
        exported.to(dtype=dtype)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_model

        save_model(
            exported,
            str(output / "model.safetensors"),
            metadata={"architecture": "AsterLM", "osp_embedding_projection": "folded"},
        )
    except Exception:
        torch.save(exported.state_dict(), output / "model.pt")
    (output / "model_config.yaml").write_text(
        yaml.safe_dump({"model": export_config.to_dict()}, sort_keys=False), encoding="utf-8"
    )
    print(f"Saved OSP-folded inference checkpoint to {output}")


if __name__ == "__main__":
    main()
