from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from asterlm.artifacts import sha256_file
from asterlm.config import DataConfig
from asterlm.training.contracts import canonical_data_config_sha256


def test_tokenizer_build_emits_corpus_bound_fertility_seal(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "text": (
                        f"Document {index} explains deterministic tokenization with "
                        "Python code, equations, and enough prose for quality filtering."
                    )
                }
            )
            + "\n"
            for index in range(20)
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "seed": 17,
                    "min_chars": 16,
                    "sources": [{"path": str(source), "weight": 1.0}],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    manifest = tmp_path / "tokenizer_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            "--data",
            str(config_path),
            "--output",
            str(tokenizer),
            "--manifest",
            str(manifest),
            "--vocab-size",
            "512",
            "--documents",
            "12",
            "--fertility-documents",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["tokenizer"]["sha256"] == sha256_file(tokenizer)
    assert payload["data_config_sha256"] == canonical_data_config_sha256(
        DataConfig.from_yaml(config_path)
    )
    assert payload["training"]["documents"] == 12
    assert payload["fertility"][0]["documents"] == 3
    assert payload["fertility"][0]["tokens"] > 0
    assert payload["fertility"][0]["sample_sha256"] != "0" * 64
