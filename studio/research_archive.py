from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
HASH_LIMIT_BYTES = 8 * 1024 * 1024
FINDINGS_PATH = Path("docs/research/findings.jsonl")

METRIC_DEFINITIONS = {
    "tokens_per_second": {
        "label": "Training throughput",
        "unit": "tokens/s",
        "direction": "higher",
        "definition": "Median measured training tokens per wall-clock second after warmup.",
    },
    "gpu_utilization": {
        "label": "GPU utilization",
        "unit": "%",
        "direction": "higher",
        "definition": "Median NVIDIA GPU busy percentage over the measured interval.",
    },
    "peak_allocated_gib": {
        "label": "Peak allocated VRAM",
        "unit": "GiB",
        "direction": "lower",
        "definition": "Peak allocator-reported CUDA memory during the measured process.",
    },
    "loss": {
        "label": "Loss",
        "unit": "nats/token",
        "direction": "lower",
        "definition": "Training or evaluation cross-entropy as recorded by the producing run.",
    },
    "time_to_common_loss": {
        "label": "Time to common quality",
        "unit": "s",
        "direction": "lower",
        "definition": "Interpolated wall time to the strongest terminal loss every compared arm is proven to reach.",
    },
    "power_w": {
        "label": "GPU power (observational)",
        "unit": "W",
        "direction": "neutral",
        "definition": "Median sampled GPU power; recorded for analysis and never used to rank candidates.",
    },
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _file_hash(path: Path, size: int) -> str | None:
    if size > HASH_LIMIT_BYTES or path.suffix.lower() in {".log", ".jsonl", ".pt", ".bin", ".safetensors"}:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name == "matrix.json":
        return "experiment_matrix"
    if name in {"experiment.json", "run_manifest.json"}:
        return "run_manifest"
    if name == "metrics.jsonl":
        return "metric_stream"
    if name == "checkpoint_manifest.json":
        return "checkpoint_manifest"
    if name.endswith("finding.json") or "findings" in path.parts:
        return "research_finding"
    if path.suffix.lower() in {".yaml", ".yml"}:
        return "configuration"
    if path.suffix.lower() == ".json":
        return "structured_result"
    if path.suffix.lower() == ".log":
        return "log"
    if path.suffix.lower() in {".md", ".txt"}:
        return "documentation"
    if path.suffix.lower() in {".pt", ".bin", ".safetensors"}:
        return "model_state"
    return "artifact"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ResearchArchive:
    """Rebuildable, revision-preserving index over AsterLM research artifacts."""

    def __init__(self, root: Path, database: Path | None = None) -> None:
        self.root = root.resolve()
        self.database = database or self.root / "data" / "aster-studio" / "research.sqlite3"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    path TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    sha256 TEXT,
                    present INTEGER NOT NULL DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    sha256 TEXT,
                    observed_at REAL NOT NULL,
                    UNIQUE(path, size_bytes, mtime_ns)
                );
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    matrix_path TEXT NOT NULL,
                    result_path TEXT,
                    name TEXT NOT NULL,
                    variant TEXT,
                    backend TEXT,
                    status TEXT NOT NULL,
                    created_utc TEXT,
                    git_commit TEXT,
                    model_path TEXT,
                    model_sha256 TEXT,
                    train_sha256 TEXT,
                    sequence_length INTEGER,
                    micro_batch_size INTEGER,
                    gradient_accumulation INTEGER,
                    global_batch_size INTEGER,
                    steps INTEGER,
                    warmup_steps INTEGER,
                    optimizer TEXT,
                    dtype TEXT,
                    precision_backend TEXT,
                    checkpoint_segment_size INTEGER,
                    gradient_checkpointing INTEGER,
                    gpu_name TEXT,
                    total_parameters INTEGER,
                    active_parameters INTEGER,
                    tokens_per_second REAL,
                    gpu_utilization REAL,
                    peak_allocated_gib REAL,
                    power_w REAL,
                    measured_tokens INTEGER,
                    measured_seconds REAL,
                    loss REAL,
                    raw_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS trials_sort ON trials(tokens_per_second DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS trials_filter ON trials(variant, backend, status);
                CREATE TABLE IF NOT EXISTS trial_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    UNIQUE(trial_id, payload_sha256)
                );
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    supersedes TEXT,
                    source_path TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS findings_created ON findings(created_utc DESC);
                """
            )
            trial_columns = {row[1] for row in db.execute("PRAGMA table_info(trials)")}
            quality_columns = {
                "quality_trial": "INTEGER NOT NULL DEFAULT 0",
                "seed": "INTEGER",
                "eval_loss": "REAL",
                "eval_loss_stdev": "REAL",
                "token_curve_auc": "REAL",
                "wall_curve_auc": "REAL",
                "equal_wall_loss": "REAL",
                "equal_flops_loss": "REAL",
                "time_to_common_loss": "REAL",
                "tokens_to_common_loss": "REAL",
                "flops_to_common_loss": "REAL",
            }
            for name, declaration in quality_columns.items():
                if name not in trial_columns:
                    db.execute(f"ALTER TABLE trials ADD COLUMN {name} {declaration}")
            db.execute(
                "INSERT OR REPLACE INTO archive_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _source_roots(self) -> list[Path]:
        return [
            self.root / "runs",
            self.root / "artifacts",
            self.root / "docs" / "experiments",
            self.root / "docs" / "research",
        ]

    def _files(self) -> Iterable[Path]:
        for source_root in self._source_roots():
            if source_root.is_file():
                yield source_root
            elif source_root.is_dir():
                yield from (path for path in source_root.rglob("*") if path.is_file())

    def reindex(self) -> dict[str, Any]:
        started = time.time()
        files = list(self._files())
        matrices: list[Path] = []
        quality_analyses: list[Path] = []
        standalone_profiles: list[tuple[Path, dict[str, Any]]] = []
        referenced_results: set[str] = set()
        artifact_revisions = 0
        trial_count = 0

        with self.connect() as db:
            db.execute("UPDATE artifacts SET present = 0")
            referenced_results.update(
                str(row[0]).replace("\\", "/").lstrip("/")
                for row in db.execute("SELECT result_path FROM trials WHERE result_path IS NOT NULL")
            )
            for path in files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = self._relative(path)
                kind = _artifact_kind(path)
                current = db.execute(
                    "SELECT size_bytes, mtime_ns, sha256, first_seen FROM artifacts WHERE path = ?",
                    (relative,),
                ).fetchone()
                unchanged = current and current[0] == stat.st_size and current[1] == stat.st_mtime_ns
                sha256 = current[2] if unchanged else _file_hash(path, stat.st_size)
                first_seen = current[3] if current else started
                db.execute(
                    """
                    INSERT INTO artifacts(path, kind, size_bytes, mtime_ns, sha256, present, first_seen, last_seen)
                    VALUES(?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        kind=excluded.kind, size_bytes=excluded.size_bytes,
                        mtime_ns=excluded.mtime_ns, sha256=excluded.sha256,
                        present=1, last_seen=excluded.last_seen
                    """,
                    (relative, kind, stat.st_size, stat.st_mtime_ns, sha256, first_seen, started),
                )
                before = db.total_changes
                db.execute(
                    """
                    INSERT OR IGNORE INTO artifact_revisions
                    (path, kind, size_bytes, mtime_ns, sha256, observed_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (relative, kind, stat.st_size, stat.st_mtime_ns, sha256, started),
                )
                artifact_revisions += int(db.total_changes > before)
                if path.name == "quality-analysis.json":
                    # Always rebuild these cheap derived rows. This also backfills
                    # quality trials after a schema upgrade even when the artifact's
                    # mtime did not change.
                    quality_analyses.append(path)
                if path.name == "matrix.json" and not unchanged:
                    matrices.append(path)
                elif not unchanged and path.suffix.lower() == ".json" and stat.st_size <= HASH_LIMIT_BYTES:
                    payload = _read_json(path)
                    if payload and isinstance(payload.get("summary"), dict) and isinstance(payload.get("system"), dict):
                        standalone_profiles.append((path, payload))

            for matrix_path in matrices:
                payload = _read_json(matrix_path)
                if not payload:
                    continue
                for trial in payload.get("trials") or []:
                    if not isinstance(trial, dict):
                        continue
                    result_path = str(trial.get("result") or "")
                    if result_path:
                        referenced_results.add(Path(result_path).as_posix().lstrip("/"))
                    self._upsert_trial(db, matrix_path, payload, trial)
                    trial_count += 1

            for result_path, payload in standalone_profiles:
                relative = self._relative(result_path)
                if relative in referenced_results or any(relative.endswith(item) for item in referenced_results):
                    continue
                synthetic_matrix = result_path.parent / "standalone-profiles"
                trial = {
                    "name": result_path.stem,
                    "variant": result_path.stem,
                    "status": payload.get("status", "unknown"),
                    "result": relative,
                    "summary": payload.get("summary"),
                }
                self._upsert_trial(db, synthetic_matrix, payload, trial, result_payload=payload)
                trial_count += 1

            for analysis_path in quality_analyses:
                payload = _read_json(analysis_path)
                if payload:
                    trial_count += self._ingest_quality_analysis(db, analysis_path, payload)

            finding_count = self._ingest_findings(db)
            db.execute("INSERT OR REPLACE INTO archive_meta(key, value) VALUES('last_indexed', ?)", (str(started),))

            counts = {
                "artifacts": db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
                "present_artifacts": db.execute("SELECT COUNT(*) FROM artifacts WHERE present=1").fetchone()[0],
                "artifact_revisions": db.execute("SELECT COUNT(*) FROM artifact_revisions").fetchone()[0],
                "trials": db.execute("SELECT COUNT(*) FROM trials").fetchone()[0],
                "trial_revisions": db.execute("SELECT COUNT(*) FROM trial_revisions").fetchone()[0],
                "findings": db.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
            }
        return {
            **counts,
            "scanned_files": len(files),
            "new_artifact_revisions": artifact_revisions,
            "processed_trials": trial_count,
            "processed_findings": finding_count,
            "indexed_at": started,
            "seconds": time.time() - started,
            "database": self._relative(self.database),
        }

    def _upsert_trial(
        self,
        db: sqlite3.Connection,
        matrix_path: Path,
        matrix: dict[str, Any],
        trial: dict[str, Any],
        *,
        result_payload: dict[str, Any] | None = None,
    ) -> None:
        matrix_relative = self._relative(matrix_path)
        name = str(trial.get("name") or trial.get("variant") or "unnamed")
        trial_id = _stable_id(matrix_relative, name)
        result_path = str(trial.get("result") or "") or None
        result_payload = result_payload or self._load_result(result_path)
        summary = trial.get("summary") if isinstance(trial.get("summary"), dict) else {}
        if result_payload and isinstance(result_payload.get("summary"), dict):
            summary = result_payload["summary"]
        protocol = matrix.get("protocol") if isinstance(matrix.get("protocol"), dict) else {}
        resolved_train = result_payload.get("resolved_train", {}) if result_payload else {}
        resolved_model = result_payload.get("resolved_model", {}) if result_payload else {}
        system = result_payload.get("system", {}) if result_payload else {}
        architecture = result_payload.get("architecture", {}) if result_payload else {}
        gpu = system.get("gpu", {}) if isinstance(system.get("gpu"), dict) else {}
        summary_gpu = summary.get("gpu", {}) if isinstance(summary.get("gpu"), dict) else {}
        memory = summary.get("final_memory", {}) if isinstance(summary.get("final_memory"), dict) else {}
        models = matrix.get("models") if isinstance(matrix.get("models"), dict) else {}
        variant = str(trial.get("variant") or name)
        model_meta = models.get(variant) if isinstance(models.get(variant), dict) else {}
        micro_batch = _as_int(protocol.get("batch", resolved_train.get("micro_batch_size")))
        accumulation = _as_int(protocol.get("accum", resolved_train.get("gradient_accumulation_steps")))
        global_batch = micro_batch * accumulation if micro_batch is not None and accumulation is not None else None
        raw = {
            "matrix": matrix_relative,
            "matrix_protocol": protocol,
            "trial": trial,
            "result": result_payload,
        }
        raw_json = _json(raw)
        raw_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        fields = (
            trial_id,
            matrix_relative,
            result_path,
            name,
            variant,
            trial.get("moe_implementation") or model_meta.get("moe_implementation") or resolved_model.get("moe_implementation"),
            str(trial.get("status") or (result_payload or {}).get("status") or "unknown"),
            matrix.get("created_utc"),
            matrix.get("git_commit") or system.get("git_commit"),
            model_meta.get("path") or (result_payload or {}).get("model_config"),
            model_meta.get("sha256"),
            (matrix.get("train_config") or {}).get("sha256") if isinstance(matrix.get("train_config"), dict) else None,
            _as_int(protocol.get("sequence", resolved_train.get("sequence_length"))),
            micro_batch,
            accumulation,
            global_batch,
            _as_int(protocol.get("steps", resolved_train.get("max_steps"))),
            _as_int(protocol.get("warmup", resolved_train.get("warmup_steps"))),
            protocol.get("optimizer_override") or resolved_train.get("optimizer"),
            protocol.get("precision_override") or resolved_train.get("dtype"),
            resolved_train.get("precision_backend"),
            _as_int(protocol.get("checkpoint_segment_size_override", resolved_model.get("checkpoint_segment_size"))),
            int(bool(resolved_model.get("gradient_checkpointing"))) if resolved_model else None,
            gpu.get("name") or system.get("nvidia_smi"),
            _as_int(architecture.get("effective_parameters") or architecture.get("trainable_parameters")),
            _as_int(architecture.get("active_parameters_estimate")),
            _as_float(summary.get("median_tokens_per_second")),
            _as_float(summary_gpu.get("median_utilization_gpu")),
            _as_float(memory.get("peak_allocated_gib")),
            _as_float(summary_gpu.get("median_power_draw")),
            _as_int(summary.get("measured_tokens")),
            _as_float(summary.get("measured_wall_time_seconds")),
            _as_float(summary.get("loss") or trial.get("loss")),
            raw_json,
            time.time(),
        )
        db.execute(
            """
            INSERT INTO trials (
                id, matrix_path, result_path, name, variant, backend, status,
                created_utc, git_commit, model_path, model_sha256, train_sha256,
                sequence_length, micro_batch_size, gradient_accumulation,
                global_batch_size, steps, warmup_steps, optimizer, dtype,
                precision_backend, checkpoint_segment_size, gradient_checkpointing,
                gpu_name, total_parameters, active_parameters, tokens_per_second,
                gpu_utilization, peak_allocated_gib, power_w, measured_tokens,
                measured_seconds, loss, raw_json, updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(id) DO UPDATE SET
                matrix_path=excluded.matrix_path, result_path=excluded.result_path,
                name=excluded.name, variant=excluded.variant, backend=excluded.backend,
                status=excluded.status, created_utc=excluded.created_utc,
                git_commit=excluded.git_commit, model_path=excluded.model_path,
                model_sha256=excluded.model_sha256, train_sha256=excluded.train_sha256,
                sequence_length=excluded.sequence_length, micro_batch_size=excluded.micro_batch_size,
                gradient_accumulation=excluded.gradient_accumulation,
                global_batch_size=excluded.global_batch_size, steps=excluded.steps,
                warmup_steps=excluded.warmup_steps, optimizer=excluded.optimizer,
                dtype=excluded.dtype, precision_backend=excluded.precision_backend,
                checkpoint_segment_size=excluded.checkpoint_segment_size,
                gradient_checkpointing=excluded.gradient_checkpointing,
                gpu_name=excluded.gpu_name, total_parameters=excluded.total_parameters,
                active_parameters=excluded.active_parameters,
                tokens_per_second=excluded.tokens_per_second,
                gpu_utilization=excluded.gpu_utilization,
                peak_allocated_gib=excluded.peak_allocated_gib,
                power_w=excluded.power_w, measured_tokens=excluded.measured_tokens,
                measured_seconds=excluded.measured_seconds, loss=excluded.loss,
                raw_json=excluded.raw_json, updated_at=excluded.updated_at
            """,
            fields,
        )
        db.execute(
            """
            INSERT OR IGNORE INTO trial_revisions(trial_id, payload_sha256, raw_json, observed_at)
            VALUES(?, ?, ?, ?)
            """,
            (trial_id, raw_hash, raw_json, time.time()),
        )

    def _ingest_quality_analysis(
        self,
        db: sqlite3.Connection,
        analysis_path: Path,
        analysis: dict[str, Any],
    ) -> int:
        campaign = _read_json(analysis_path.with_name("quality-campaign.json")) or {}
        run_records = [row for row in analysis.get("runs", []) if isinstance(row, dict)]
        count = 0
        representative_by_identity: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for record in run_records:
            run_dir = analysis_path.parent / str(record.get("run_dir") or "")
            experiment_path = run_dir / "experiment.json"
            experiment = _read_json(experiment_path) or {}
            identity = f"{record.get('candidate_id')}:{record.get('execution_variant')}"
            if experiment:
                representative_by_identity.setdefault(identity, (record, experiment))
            self._upsert_quality_trial(
                db,
                analysis_path=analysis_path,
                campaign=campaign,
                record=record,
                experiment=experiment,
                aggregate=None,
            )
            count += 1

        for identity, aggregate in (analysis.get("candidates") or {}).items():
            if not isinstance(aggregate, dict):
                continue
            representative = representative_by_identity.get(str(identity), ({}, {}))
            self._upsert_quality_trial(
                db,
                analysis_path=analysis_path,
                campaign=campaign,
                record=representative[0],
                experiment=representative[1],
                aggregate=aggregate,
            )
            count += 1
        return count

    def _upsert_quality_trial(
        self,
        db: sqlite3.Connection,
        *,
        analysis_path: Path,
        campaign: dict[str, Any],
        record: dict[str, Any],
        experiment: dict[str, Any],
        aggregate: dict[str, Any] | None,
    ) -> None:
        matrix_relative = self._relative(analysis_path)
        candidate = str(
            (aggregate or {}).get("candidate_id") or record.get("candidate_id") or "candidate"
        )
        variant = str(
            (aggregate or {}).get("execution_variant")
            or record.get("execution_variant")
            or "quality"
        )
        seed = None if aggregate is not None else _as_int(record.get("seed"))
        identity = f"{candidate}:{variant}"
        name = f"{identity} aggregate" if aggregate is not None else f"{identity} seed-{seed}"
        trial_id = _stable_id(matrix_relative, name)
        train = ((experiment.get("train") or {}).get("config") or {})
        model = experiment.get("model") or {}
        architecture = model.get("architecture") or {}
        environment = experiment.get("environment") or {}
        gpu = environment.get("gpu") or {}
        code = experiment.get("code") or {}
        campaign_runs = campaign.get("runs") or {}
        run_key = f"{seed}:{candidate}:{variant}" if seed is not None else ""
        campaign_run = campaign_runs.get(run_key) if isinstance(campaign_runs, dict) else {}
        campaign_run = campaign_run if isinstance(campaign_run, dict) else {}
        complete = int((aggregate or {}).get("complete_seed_count") or 0)
        expected = int((aggregate or {}).get("expected_seed_count") or 0)
        status = (
            ("ok" if expected > 0 and complete == expected else "partial")
            if aggregate is not None
            else str(record.get("status") or experiment.get("status") or "missing")
        )
        tokens_per_update = (
            (_as_int(train.get("sequence_length")) or 0)
            * (_as_int(train.get("micro_batch_size")) or 0)
            * (_as_int(train.get("gradient_accumulation_steps")) or 0)
        )
        curve = record.get("learning_curve") if isinstance(record.get("learning_curve"), list) else []
        steps = max((_as_int(point.get("step")) or 0 for point in curve), default=None)
        eval_loss = _as_float(
            (aggregate or {}).get("final_eval_loss_mean")
            if aggregate is not None
            else record.get("eval_main_loss")
        )
        raw = {
            "analysis": matrix_relative,
            "campaign_type": campaign.get("campaign_type", "architecture_quality"),
            "comparison_contract": campaign.get("comparison_contract"),
            "candidate": candidate,
            "variant": variant,
            "seed": seed,
            "record": record,
            "aggregate": aggregate,
            "experiment": experiment,
        }
        raw_json = _json(raw)
        experiment_relative = None
        if aggregate is None and record.get("run_dir"):
            experiment_relative = self._relative(
                analysis_path.parent / str(record["run_dir"]) / "experiment.json"
            )
        values = {
            "id": trial_id,
            "matrix_path": matrix_relative,
            "result_path": (
                matrix_relative
                if aggregate is not None
                else experiment_relative
            ),
            "name": name,
            "variant": variant,
            "backend": variant,
            "status": status,
            "created_utc": experiment.get("started_at_utc"),
            "git_commit": code.get("git_commit")
            or (campaign.get("source_provenance") or {}).get("git_commit"),
            "model_path": campaign_run.get("model_config") or campaign.get("model"),
            "model_sha256": campaign_run.get("model_config_sha256")
            or model.get("config_sha256")
            or campaign.get("model_sha256"),
            "train_sha256": (experiment.get("train") or {}).get("config_sha256"),
            "sequence_length": _as_int(train.get("sequence_length")),
            "micro_batch_size": _as_int(train.get("micro_batch_size")),
            "gradient_accumulation": _as_int(train.get("gradient_accumulation_steps")),
            "global_batch_size": (
                (_as_int(train.get("micro_batch_size")) or 0)
                * (_as_int(train.get("gradient_accumulation_steps")) or 0)
            )
            or None,
            "steps": steps,
            "warmup_steps": _as_int(train.get("warmup_steps")),
            "optimizer": train.get("optimizer"),
            "dtype": train.get("dtype"),
            "precision_backend": train.get("precision_backend"),
            "checkpoint_segment_size": _as_int(
                (model.get("config") or {}).get("checkpoint_segment_size")
            ),
            "gradient_checkpointing": (
                int(bool((model.get("config") or {}).get("gradient_checkpointing")))
                if model
                else None
            ),
            "gpu_name": gpu.get("name") or environment.get("nvidia_smi"),
            "total_parameters": _as_int(
                architecture.get("effective_parameters")
                or architecture.get("trainable_parameters")
            ),
            "active_parameters": _as_int(architecture.get("active_parameters_estimate")),
            "tokens_per_second": _as_float(
                (aggregate or {}).get("median_training_tokens_per_second")
                if aggregate is not None
                else record.get("median_training_tokens_per_second")
            ),
            "gpu_utilization": _as_float(
                (aggregate or {}).get("mean_gpu_util_percent")
                if aggregate is not None
                else record.get("mean_gpu_util_percent")
            ),
            "peak_allocated_gib": _as_float(
                (aggregate or {}).get("peak_vram_gib")
                if aggregate is not None
                else record.get("peak_vram_gib")
            ),
            "power_w": None,
            "measured_tokens": _as_int(record.get("tokens_seen"))
            or (tokens_per_update * steps if steps is not None else None),
            "measured_seconds": _as_float(record.get("wall_clock_total_seconds")),
            "loss": eval_loss,
            "raw_json": raw_json,
            "updated_at": time.time(),
            "quality_trial": 1,
            "seed": seed,
            "eval_loss": eval_loss,
            "eval_loss_stdev": _as_float((aggregate or {}).get("final_eval_loss_stdev")),
            "token_curve_auc": _as_float((aggregate or {}).get("token_curve_auc_mean")),
            "wall_curve_auc": _as_float((aggregate or {}).get("wall_curve_auc_mean")),
            "equal_wall_loss": _as_float((aggregate or {}).get("equal_wall_loss_mean")),
            "equal_flops_loss": _as_float(
                (aggregate or {}).get("equal_active_flops_loss_mean")
            ),
            "time_to_common_loss": _as_float(
                (aggregate or {}).get("time_to_common_loss_seconds_mean")
            ),
            "tokens_to_common_loss": _as_float(
                (aggregate or {}).get("tokens_to_common_loss_mean")
            ),
            "flops_to_common_loss": _as_float(
                (aggregate or {}).get("active_flops_to_common_loss_mean")
            ),
        }
        columns = list(values)
        assignments = ",".join(f"{column}=excluded.{column}" for column in columns if column != "id")
        db.execute(
            f"INSERT INTO trials ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}",
            tuple(values[column] for column in columns),
        )
        raw_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        db.execute(
            """
            INSERT OR IGNORE INTO trial_revisions(trial_id, payload_sha256, raw_json, observed_at)
            VALUES(?, ?, ?, ?)
            """,
            (trial_id, raw_hash, raw_json, time.time()),
        )

    def _load_result(self, value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                candidate = self.root / candidate.resolve().relative_to(self.root)
            except ValueError:
                return None
        else:
            candidate = self.root / candidate
        return _read_json(candidate)

    def _ingest_findings(self, db: sqlite3.Connection) -> int:
        path = self.root / FINDINGS_PATH
        if not path.is_file():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    finding = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(finding, dict):
                    continue
                finding_id = str(finding.get("id") or _stable_id(str(FINDINGS_PATH), str(line_number), line))
                raw_json = _json(finding)
                db.execute(
                    """
                    INSERT INTO findings VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        created_utc=excluded.created_utc, title=excluded.title,
                        summary=excluded.summary, status=excluded.status,
                        tags_json=excluded.tags_json, evidence_json=excluded.evidence_json,
                        supersedes=excluded.supersedes, source_path=excluded.source_path,
                        raw_json=excluded.raw_json
                    """,
                    (
                        finding_id,
                        str(finding.get("created_utc") or ""),
                        str(finding.get("title") or finding_id),
                        str(finding.get("summary") or ""),
                        str(finding.get("status") or "observed"),
                        _json(finding.get("tags") or []),
                        _json(finding.get("evidence") or []),
                        finding.get("supersedes"),
                        FINDINGS_PATH.as_posix(),
                        raw_json,
                    ),
                )
                count += 1
        return count

    def summary(self) -> dict[str, Any]:
        with self.connect() as db:
            def scalar(sql: str, args: tuple[Any, ...] = ()) -> Any:
                return db.execute(sql, args).fetchone()[0]

            last = db.execute("SELECT value FROM archive_meta WHERE key='last_indexed'").fetchone()
            fastest = db.execute(
                """
                SELECT id, name, variant, tokens_per_second, gpu_utilization, matrix_path
                FROM trials WHERE status='ok' AND tokens_per_second IS NOT NULL
                ORDER BY tokens_per_second DESC LIMIT 1
                """
            ).fetchone()
            backends = [row[0] for row in db.execute("SELECT DISTINCT backend FROM trials WHERE backend IS NOT NULL ORDER BY backend")]
            statuses = [row[0] for row in db.execute("SELECT DISTINCT status FROM trials ORDER BY status")]
            return {
                "schema_version": SCHEMA_VERSION,
                "database": self._relative(self.database),
                "last_indexed": float(last[0]) if last else None,
                "artifacts": scalar("SELECT COUNT(*) FROM artifacts"),
                "present_artifacts": scalar("SELECT COUNT(*) FROM artifacts WHERE present=1"),
                "artifact_revisions": scalar("SELECT COUNT(*) FROM artifact_revisions"),
                "trials": scalar("SELECT COUNT(*) FROM trials"),
                "successful_trials": scalar("SELECT COUNT(*) FROM trials WHERE status='ok'"),
                "trial_revisions": scalar("SELECT COUNT(*) FROM trial_revisions"),
                "findings": scalar("SELECT COUNT(*) FROM findings"),
                "fastest_observed": dict(fastest) if fastest else None,
                "backends": backends,
                "statuses": statuses,
                "metric_definitions": METRIC_DEFINITIONS,
                "retention": "unbounded; no automatic row or artifact deletion",
            }

    def trials(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        query: str = "",
        backend: str = "",
        status: str = "",
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        clauses: list[str] = []
        args: list[Any] = []
        if query:
            clauses.append("(name LIKE ? OR variant LIKE ? OR matrix_path LIKE ? OR model_path LIKE ?)")
            needle = f"%{query}%"
            args.extend([needle] * 4)
        if backend:
            clauses.append("backend = ?")
            args.append(backend)
        if status:
            clauses.append("status = ?")
            args.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            total = db.execute(f"SELECT COUNT(*) FROM trials{where}", args).fetchone()[0]
            rows = db.execute(
                f"""
                SELECT * FROM trials{where}
                ORDER BY COALESCE(tokens_per_second, -1) DESC, updated_at DESC
                LIMIT ? OFFSET ?
                """,
                [*args, limit, offset],
            ).fetchall()
        return {"rows": [self._public_trial(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    @staticmethod
    def _public_trial(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        raw_json = payload.pop("raw_json", None)
        try:
            raw = json.loads(raw_json) if raw_json else {}
        except (TypeError, json.JSONDecodeError):
            raw = {}
        stability = raw.get("aggregate") or raw.get("record") or {}
        if isinstance(stability, dict):
            payload.update(
                {
                    "run_survival_percent": (
                        _as_float(stability.get("run_survival_rate")) * 100
                        if _as_float(stability.get("run_survival_rate")) is not None
                        else (100.0 if stability.get("run_survived") else None)
                    ),
                    "gradient_nonfinite_count": _as_int(
                        stability.get("gradient_nonfinite_count")
                    ),
                    "training_loss_nonfinite_count": _as_int(
                        stability.get("training_loss_nonfinite_count")
                    ),
                    "gradient_norm_p95": _as_float(
                        stability.get("gradient_norm_p95_mean")
                        if stability.get("gradient_norm_p95_mean") is not None
                        else stability.get("gradient_norm_p95")
                    ),
                    "gradient_clip_percent": (
                        _as_float(stability.get("gradient_clip_fraction_mean")) * 100
                        if _as_float(stability.get("gradient_clip_fraction_mean")) is not None
                        else (
                            _as_float(stability.get("gradient_clip_fraction")) * 100
                            if _as_float(stability.get("gradient_clip_fraction")) is not None
                            else None
                        )
                    ),
                    "loss_upward_jump_gt_0_5_count": _as_int(
                        stability.get("loss_upward_jump_gt_0_5_count")
                    ),
                    "parameter_rms_drift_percent": (
                        _as_float(stability.get("parameter_global_rms_relative_drift_mean"))
                        * 100
                        if _as_float(
                            stability.get("parameter_global_rms_relative_drift_mean")
                        )
                        is not None
                        else (
                            _as_float(stability.get("parameter_global_rms_relative_drift"))
                            * 100
                            if _as_float(
                                stability.get("parameter_global_rms_relative_drift")
                            )
                            is not None
                            else None
                        )
                    ),
                    "optimizer_wall_percent": (
                        _as_float(stability.get("optimizer_wall_fraction_mean")) * 100
                        if _as_float(stability.get("optimizer_wall_fraction_mean")) is not None
                        else None
                    ),
                    "muon_relative_update_rms": _as_float(
                        stability.get("muon_relative_update_rms_mean")
                    ),
                }
            )
        return payload

    def compare(self, ids: list[str]) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(ids))[:24]
        if not unique_ids:
            return {"trials": [], "dimensions": [], "strictly_comparable": False}
        placeholders = ",".join("?" for _ in unique_ids)
        with self.connect() as db:
            rows = db.execute(f"SELECT * FROM trials WHERE id IN ({placeholders})", unique_ids).fetchall()
        by_id = {row["id"]: row for row in rows}
        ordered = [by_id[item] for item in unique_ids if item in by_id]
        campaign_types: set[str] = set()
        declared_treatments: set[str] = set()
        matrix_paths = {row["matrix_path"] for row in ordered}
        for row in ordered:
            try:
                raw = json.loads(row["raw_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            campaign_type = raw.get("campaign_type")
            if campaign_type:
                campaign_types.add(str(campaign_type))
            contract = raw.get("comparison_contract")
            if isinstance(contract, dict):
                treatments = contract.get("treatment_fields")
                if isinstance(treatments, list):
                    declared_treatments.update(str(field) for field in treatments)
        same_campaign = len(matrix_paths) == 1 and len(campaign_types) <= 1
        treatment_fields = declared_treatments if same_campaign else set()
        if same_campaign and campaign_types == {"optimizer_quality"}:
            # Backwards-compatible interpretation for optimizer campaigns created
            # before the explicit comparison contract was added.
            treatment_fields.update({"optimizer", "warmup_steps"})
        dimensions = []
        strict_fields = [
            ("sequence_length", "sequence length"),
            ("global_batch_size", "global batch"),
            ("steps", "measured steps"),
            ("warmup_steps", "warmup steps"),
            ("optimizer", "optimizer"),
            ("dtype", "dtype"),
            ("precision_backend", "precision backend"),
            ("gpu_name", "GPU"),
        ]
        for field, label in strict_fields:
            values = [row[field] for row in ordered]
            dimensions.append(
                {
                    "field": field,
                    "label": label,
                    "match": len(set(values)) <= 1,
                    "values": values,
                    "role": "treatment" if field in treatment_fields else "control",
                }
            )
        control_dimensions = [item for item in dimensions if item["role"] == "control"]
        treatment_dimensions = [
            item for item in dimensions if item["role"] == "treatment" and not item["match"]
        ]
        controls_match = len(ordered) > 1 and all(
            item["match"] for item in control_dimensions
        )
        comparison_kind = (
            "protocol_mismatch"
            if not controls_match
            else "controlled_treatment"
            if treatment_dimensions
            else "matched_protocol"
        )
        return {
            "trials": [self._public_trial(row) for row in ordered],
            "dimensions": dimensions,
            "strictly_comparable": controls_match,
            "comparison_kind": comparison_kind,
            "campaign_type": next(iter(campaign_types), None),
            "treatment_fields": sorted(treatment_fields),
            "same_model": len({row["model_sha256"] for row in ordered}) <= 1,
            "note": (
                "Intentional treatment differences are separated from fixed protocol controls. "
                "Energy is displayed for analysis only and never participates in candidate ranking."
            ),
        }

    def findings(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            rows = db.execute(
                "SELECT * FROM findings ORDER BY created_utc DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.pop("tags_json"))
            item["evidence"] = json.loads(item.pop("evidence_json"))
            item.pop("raw_json", None)
            output.append(item)
        return {"rows": output, "total": total, "limit": limit, "offset": offset}
