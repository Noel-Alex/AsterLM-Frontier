from __future__ import annotations

import json
import os
from pathlib import Path

from asterlm.triton_cache import quarantine_invalid_triton_json


def test_stale_invalid_json_is_quarantined_but_valid_and_recent_files_remain(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    stale = cache / "broken" / "kernel.json"
    valid = cache / "valid" / "kernel.json"
    recent = cache / "recent" / "kernel.json"
    for path in (stale, valid, recent):
        path.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("", encoding="utf-8")
    valid.write_text(json.dumps({"configs_timings": {}}), encoding="utf-8")
    recent.write_text("", encoding="utf-8")
    os.utime(stale, (100.0, 100.0))
    os.utime(valid, (100.0, 100.0))
    os.utime(recent, (950.0, 950.0))

    repairs = quarantine_invalid_triton_json(
        cache,
        minimum_age_seconds=300.0,
        now=1000.0,
        force_rescan=True,
    )

    assert len(repairs) == 1
    assert repairs[0]["source"] == str(stale.resolve())
    assert not stale.exists()
    assert Path(repairs[0]["quarantine"]).read_text(encoding="utf-8") == ""
    assert valid.is_file()
    assert recent.is_file()
