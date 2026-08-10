from __future__ import annotations

import json

from studio import server


def test_diagnostic_matrices_summarize_nested_runs(tmp_path, monkeypatch):
    matrix_dir = tmp_path / "runs" / "frontier" / "matched-screen"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "matrix.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-08-10T00:00:00+00:00",
                "completed_utc": "2026-08-10T00:01:00+00:00",
                "git_commit": "abc123",
                "protocol": {"repetitions": 1},
                "trials": [
                    {"name": "dense", "status": "ok", "returncode": 0},
                    {"name": "moe", "status": "oom", "returncode": 2},
                ],
                "aggregate": {
                    "dense": {
                        "successful_repetitions": 1,
                        "median_tokens_per_second": 1234.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "ROOT", tmp_path)

    rows = server.diagnostic_matrices()

    assert len(rows) == 1
    assert rows[0]["name"] == "matched-screen"
    assert rows[0]["status"] == "completed"
    assert rows[0]["successful_trials"] == 1
    assert rows[0]["failures"] == [{"name": "moe", "status": "oom", "returncode": 2}]
    assert rows[0]["aggregate"]["dense"]["median_tokens_per_second"] == 1234.0
