from __future__ import annotations

import json
from pathlib import Path

import yaml

from asterlm.data.resumable import RetryPolicy, clean_uncommitted_outputs, restore_dataset_cursor


class FakeDataset:
    def __init__(self) -> None:
        self.loaded = None
        self.skipped = None

    def load_state_dict(self, state):
        self.loaded = state

    def skip(self, count):
        self.skipped = count
        return self


def test_cursor_resume_preferred_over_record_skip() -> None:
    dataset = FakeDataset()
    restored = restore_dataset_cursor(dataset, {"shard": 3, "row": 7}, fallback_skip=999)
    assert restored is dataset
    assert dataset.loaded == {"shard": 3, "row": 7}
    assert dataset.skipped is None


def test_record_skip_is_legacy_fallback() -> None:
    class Legacy:
        def __init__(self):
            self.skipped = None

        def skip(self, count):
            self.skipped = count
            return self

    dataset = Legacy()
    restored = restore_dataset_cursor(dataset, None, fallback_skip=123)
    assert restored is dataset
    assert dataset.skipped == 123


def test_orphan_cleanup_keeps_only_committed_files(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for name in (
        "demo-00000.jsonl.zst",
        "demo-00001.jsonl.zst",
        "demo-00002.jsonl.zst.partial",
        "cursor-00000001.pkl",
        "cursor-00000002.pkl",
        "cursor-00000003.pkl.tmp",
    ):
        (root / name).write_bytes(b"x")

    removed = clean_uncommitted_outputs(root, "demo", next_shard_index=1, checkpoint_id=1)
    assert (root / "demo-00000.jsonl.zst").exists()
    assert not (root / "demo-00001.jsonl.zst").exists()
    assert not (root / "demo-00002.jsonl.zst.partial").exists()
    assert (root / "cursor-00000001.pkl").exists()
    assert not (root / "cursor-00000002.pkl").exists()
    assert any("demo-00001" in path for path in removed)


def test_retry_policy_supports_unlimited_mode() -> None:
    assert RetryPolicy(max_retries=0).permits(100000)
    assert RetryPolicy(max_retries=3).permits(3)
    assert not RetryPolicy(max_retries=3).permits(4)


def test_dclm_uses_official_parquet_mirror() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("corpus_pilot_500m.yaml", "corpus_frontier_16b.yaml", "corpus_main_12b.yaml"):
        raw = yaml.safe_load((root / "configs" / "corpus" / name).read_text(encoding="utf-8"))
        dclm = next(item for item in raw["corpus"]["sources"] if item["id"] == "dclm")
        assert dclm["path"] == "mlfoundations/dclm-baseline-1.0-parquet"
        assert "text" in dclm["columns"]


def test_pilot_progressively_expands_frontier_output() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = yaml.safe_load((root / "configs/corpus/corpus_pilot_500m.yaml").read_text(encoding="utf-8"))
    frontier = yaml.safe_load((root / "configs/corpus/corpus_frontier_16b.yaml").read_text(encoding="utf-8"))
    assert pilot["corpus"]["output_dir"] == frontier["corpus"]["output_dir"]


def _load_script_module(name: str):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_corpus_signature_allows_target_expansion_but_not_filter_changes() -> None:
    module = _load_script_module("materialize_corpus")
    small = module.CorpusSource(id="x", path="repo", target_tokens=100, min_chars=10)
    large = module.CorpusSource(id="x", path="repo", target_tokens=1000, min_chars=10)
    changed = module.CorpusSource(id="x", path="repo", target_tokens=1000, min_chars=20)
    assert module.source_signature(small) == module.source_signature(large)
    assert module.source_signature(small) != module.source_signature(changed)


def test_record_completion_supports_full_split_mode() -> None:
    module = _load_script_module("materialize_hf_records")
    full = module.RecordSource(id="mmlu", path="cais/mmlu", split="test", target_records=None)
    capped = module.RecordSource(id="mmlu", path="cais/mmlu", split="test", target_records=15000)
    state = {"written": 14042, "source_exhausted": True}
    assert module.source_complete(full, state)
    assert not module.source_complete(capped, state)


def test_complete_benchmark_splits_do_not_use_impossible_caps() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (root / "configs/corpus/decontamination_benchmarks.yaml").read_text(encoding="utf-8")
    )
    assert all(source["target_records"] is None for source in raw["records"]["sources"])


def test_old_mmlu_state_migrates_to_complete_split_without_redownload(tmp_path: Path) -> None:
    module = _load_script_module("materialize_hf_records")
    old_source = module.RecordSource(
        id="mmlu",
        path="cais/mmlu",
        name="all",
        split="test",
        target_records=15000,
        keep_fields=["question", "choices", "answer", "subject"],
        shuffle_seed=2000,
    )
    new_source = module.RecordSource(
        id="mmlu",
        path="cais/mmlu",
        name="all",
        split="test",
        target_records=None,
        keep_fields=["question", "choices", "answer", "subject"],
        shuffle_seed=2000,
    )
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "source": module.asdict(old_source),
                "source_signature": module.legacy_signature(old_source),
                "seen": 14042,
                "written": 14042,
                "source_exhausted": True,
                "complete": False,
            }
        ),
        encoding="utf-8",
    )
    state = module.load_state(path, new_source)
    assert state["version"] == 3
    assert module.source_complete(new_source, state)


def test_project_local_hf_cache_preserves_global_login_token(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    module = _load_script_module("download_data")
    global_home = tmp_path / "global-hf"
    global_home.mkdir()
    token = global_home / "token"
    token.write_text("hf_test_token", encoding="utf-8")
    project_home = tmp_path / "project-cache"

    monkeypatch.setenv("HF_HOME", str(global_home))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    env = module.build_environment(SimpleNamespace(hf_home=str(project_home), network_mode="low"))

    assert env["HF_HOME"] == str(project_home.resolve())
    assert env["HF_HUB_CACHE"] == str(project_home.resolve() / "hub")
    assert env["HF_TOKEN_PATH"] == str(token.resolve())


def test_explicit_hf_token_path_is_not_replaced(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    module = _load_script_module("download_data")
    explicit = tmp_path / "explicit-token"
    explicit.write_text("hf_explicit", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(explicit))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    env = module.build_environment(
        SimpleNamespace(hf_home=str(tmp_path / "project-cache"), network_mode="low")
    )
    assert env["HF_TOKEN_PATH"] == str(explicit)


def test_checkout_guard_rejects_another_checkout_virtualenv(tmp_path: Path) -> None:
    module = _load_script_module("download_data")
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    (other / ".venv").mkdir(parents=True)

    errors = module.execution_context_errors(
        repo_root=repo,
        cwd=repo,
        virtual_env=str(other / ".venv"),
        allow_external_venv=False,
    )
    assert any("different checkout" in error for error in errors)
    assert not module.execution_context_errors(
        repo_root=repo,
        cwd=repo,
        virtual_env=str(other / ".venv"),
        allow_external_venv=True,
    )
