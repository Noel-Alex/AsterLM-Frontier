from __future__ import annotations

import json
from pathlib import Path

from studio import server


def test_tail_lines_reads_only_the_requested_suffix(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(f"{index}\n" for index in range(10_000)), encoding="utf-8")

    assert server.tail_lines(path, 3) == ["9997\n", "9998\n", "9999\n"]


def test_nested_quality_run_is_visible_in_live_status(
    monkeypatch, tmp_path: Path
) -> None:
    run = (
        tmp_path
        / "runs"
        / "architecture-campaign"
        / "decision"
        / "seed-1337"
        / "moe"
        / "bf16"
    )
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "run_id": "live-moe",
                "status": "running",
                "stage": "pretrain",
                "completion_fraction": 0.25,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 10,
                "tokens_seen": 163_840,
                "tokens_per_second": 3_000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "ROOT", tmp_path)
    server._RUN_PATH_CACHE.update({"updated": 0.0, "paths": []})

    rows = server.runs_status()

    assert rows[0]["run_id"] == "live-moe"
    assert rows[0]["latest"]["tokens_seen"] == 163_840
    assert rows[0]["status"] == "running"
