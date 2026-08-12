#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from torch.utils.data import DataLoader

from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.generation.hub_checkpoint import resolve_hub_auto_resume
from asterlm.training import Trainer
from asterlm.training.checkpoint import resolve_checkpoint
from studio.stateful_data import StatefulLocalPackedDataset


class StudioStopRequested(BaseException):
    """Raised only at a safe training-update boundary."""

    asterlm_status = "interrupted_user"


def normalize_studio_data_state(state: dict[str, Any]) -> dict[str, Any]:
    """Translate the short-lived Studio envelope to the canonical trainer schema."""

    if "schema_version" in state:
        return state
    if int(state.get("version", -1)) == 1:
        return {"schema_version": 1, **{key: value for key, value in state.items() if key != "version"}}
    return state


class StudioTrainer(Trainer):
    """Trainer overlay with fast local data cursors and graceful stop checkpoints."""

    def __init__(
        self,
        model_config: AsterConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
        *,
        mode: str,
        initial_checkpoint: str | None,
    ) -> None:
        selected_sources = (
            list(data_config.sources)
            + list(data_config.validation_sources)
        )
        self._studio_fast_data = (
            mode == "pretrain"
            and bool(data_config.sources)
            and all(Path(source.path).exists() for source in selected_sources)
            and os.getenv("ASTER_STUDIO_FAST_DATA_RESUME", "1") != "0"
        )
        self._studio_train_dataset: StatefulLocalPackedDataset | None = None
        self._studio_validation_dataset: StatefulLocalPackedDataset | None = None

        if self._studio_fast_data:
            # Worker-prefetch state is not serializable. The cleaned local corpus
            # is fast enough that exact, O(1)-ish resume is more valuable than
            # worker-level input prefetch.
            train_config.num_workers = 0
            train_config.prefetch_factor = None

        super().__init__(
            model_config,
            train_config,
            data_config,
            mode=mode,
            initial_checkpoint=initial_checkpoint,
        )

    def _build_loader(self, validation: bool) -> DataLoader:
        if not getattr(self, "_studio_fast_data", False):
            return super()._build_loader(validation)

        dataset = StatefulLocalPackedDataset(
            self.tokenizer,
            self.data_config,
            self.train_config.sequence_length,
            validation=validation,
            ignore_index=self.train_config.ignore_index,
        )
        if validation:
            self._studio_validation_dataset = dataset
        else:
            self._studio_train_dataset = dataset

        return DataLoader(
            dataset=dataset,
            batch_size=self.train_config.micro_batch_size,
            num_workers=0,
            pin_memory=self.train_config.pin_memory and self.device.type == "cuda",
        )

    def _restore_training_data_position(self) -> None:
        if not getattr(self, "_studio_fast_data", False):
            return super()._restore_training_data_position()

        assert self.train_config.resume is not None
        checkpoint = resolve_checkpoint(self.train_config.resume)
        cursor_path = checkpoint / "data_state.pt"
        if not cursor_path.is_file():
            print(
                "Studio data cursor is absent in this older checkpoint; "
                "falling back to the repository's exact replay restore.",
                flush=True,
            )
            return super()._restore_training_data_position()

        import torch

        state = torch.load(cursor_path, map_location="cpu", weights_only=False)
        state = normalize_studio_data_state(state)
        if int(state.get("schema_version", -1)) != 1:
            raise RuntimeError(
                "Unsupported studio_data_state schema_version="
                f"{state.get('schema_version')}"
            )
        if int(state.get("tokens_seen", -1)) != int(self.tokens_seen):
            raise RuntimeError(
                "Studio data cursor tokens_seen does not match trainer_state.pt"
            )
        if int(state.get("step", -1)) != int(self.step):
            raise RuntimeError(
                "Studio data cursor step does not match trainer_state.pt"
            )

        if self._studio_train_dataset is None:
            raise RuntimeError("Studio train dataset was not initialized")
        self._studio_train_dataset.load_state_dict(state["train"])
        # Recreate the DataLoader iterator after loading the dataset state.
        self.train_iterator = iter(self.train_loader)

        validation_state = state.get("validation")
        if (
            validation_state is not None
            and self._studio_validation_dataset is not None
            and self.validation_loader is not None
        ):
            self._studio_validation_dataset.load_state_dict(validation_state)
            self.validation_iterator = iter(self.validation_loader)

        cursors = state["train"].get("cursors", [])
        cursor_summary = ", ".join(
            f"s{i}:file={item.get('file_index')} row={item.get('record_index'):,}"
            for i, item in enumerate(cursors)
        )
        print(
            "Studio fast data resume restored packed buffer + source/RNG state; "
            + cursor_summary,
            flush=True,
        )

    def _restore_training_data_state(self, state: dict[str, Any]) -> None:
        """Restore both canonical and legacy Studio cursor envelopes exactly."""

        normalized = normalize_studio_data_state(state)
        super()._restore_training_data_state(normalized)
        train_state = normalized.get("train") or {}
        cursors = train_state.get("cursors", [])
        cursor_summary = ", ".join(
            f"s{i}:file={item.get('file_index')} row={int(item.get('record_index', 0)):,}"
            for i, item in enumerate(cursors)
        )
        print(
            "Studio fast data resume restored packed buffer + source/RNG state; "
            + cursor_summary,
            flush=True,
        )

    def _studio_data_state(self) -> dict[str, Any] | None:
        if not self._studio_fast_data or self._studio_train_dataset is None:
            return None
        return {
            "schema_version": 1,
            "step": int(self.step),
            "tokens_seen": int(self.tokens_seen),
            "train": self._studio_train_dataset.state_dict(),
            "validation": (
                self._studio_validation_dataset.state_dict()
                if self._studio_validation_dataset is not None
                else None
            ),
        }

    def _training_data_state(self) -> dict[str, Any]:
        if not self._studio_fast_data:
            return super()._training_data_state()
        state = self._studio_data_state()
        if state is None:  # pragma: no cover - guarded by _studio_fast_data
            raise RuntimeError("Studio fast data state is unexpectedly unavailable")
        return state

    def _save(
        self,
        reason: str,
        *,
        permanent: bool = False,
        tag: str | None = None,
    ) -> Path:
        return super()._save(reason, permanent=permanent, tag=tag)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AsterLM Studio trainer with graceful stop + checkpointed local data cursors"
    )
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--data", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", default=None)
    group.add_argument(
        "--init-checkpoint",
        "--checkpoint",
        dest="init_checkpoint",
        default=None,
    )
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument(
        "--remote-durable",
        action="store_true",
        help="Require persistent output plus verified full-state Hub upload at every save",
    )
    args = parser.parse_args()

    train_config = TrainConfig.from_yaml(args.train)
    model_config = AsterConfig.from_yaml(args.model)
    if args.resume:
        train_config.resume = args.resume
    if args.hub_repo:
        train_config.hub_repo_id = args.hub_repo
    if args.remote_durable:
        if not args.hub_repo:
            raise ValueError("--remote-durable requires --hub-repo")
        remote_run_root = Path(os.environ.get("ASTERLM_REMOTE_RUN_ROOT", "/opt/aster/runs"))
        train_config.output_dir = str(remote_run_root / Path(train_config.output_dir).name)
        train_config.checkpoint_interval_minutes = 5.0
        train_config.hub_private = True
        train_config.hub_upload_every_save = True
        train_config.hub_upload_milestones = True
        train_config.hub_upload_final = True
        train_config.hub_upload_on_stop = True
        train_config.hub_auto_resume_latest = True
        train_config.hub_include_optimizer = True
        train_config.hub_fail_on_error = True
        train_config.hub_async_upload = True
        train_config.hub_max_pending_uploads = 2

    if (
        train_config.hub_auto_resume_latest
        and train_config.hub_repo_id
        and not args.init_checkpoint
    ):
        resume, record = resolve_hub_auto_resume(
            output_dir=train_config.output_dir,
            repo_id=train_config.hub_repo_id,
            revision=train_config.hub_revision,
            local_checkpoint=train_config.resume,
            requested_model=model_config,
        )
        if resume is not None:
            train_config.resume = str(resume)
        print(f"Hub/local auto-resume: {record}", flush=True)

    trainer = StudioTrainer(
        model_config,
        train_config,
        DataConfig.from_yaml(args.data),
        mode=args.mode,
        initial_checkpoint=args.init_checkpoint,
    )
    print(trainer.model.architecture_summary(), flush=True)
    if trainer._studio_fast_data:
        print(
            "AsterLM Studio fast local-data resume: ENABLED "
            "(source cursors + mixture/FIM RNG + residual pack buffer)",
            flush=True,
        )

    stop_requested = threading.Event()
    stop_diagnostic_setting: bool | None = None
    train_batches_since_launch = 0
    accumulation = max(
        1,
        int(train_config.gradient_accumulation_steps),
    )
    original_next_batch = trainer._next_batch

    def request_stop(signum, frame):
        del frame
        if not stop_requested.is_set():
            print(
                f"\nAsterLM Studio received signal {signum}; "
                "finishing the current optimizer update before checkpointing...",
                flush=True,
            )
            stop_requested.set()
        else:
            print(
                "\nAnother stop signal was received. Studio is still waiting "
                "for a safe optimizer-update boundary.",
                flush=True,
            )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def controlled_next_batch(self, validation: bool = False):
        nonlocal train_batches_since_launch, stop_diagnostic_setting
        if (
            not validation
            and stop_requested.is_set()
            and train_batches_since_launch % accumulation == 0
        ):
            # Trainer.train() treats every BaseException as a failure and would
            # otherwise write a misleading "failure" bundle. Suppress that one
            # bundle only; the outer Studio stop handler restores the setting and
            # writes a normal studio-stop checkpoint/diagnostic instead.
            stop_diagnostic_setting = bool(self.train_config.save_diagnostic_bundle)
            self.train_config.save_diagnostic_bundle = False
            raise StudioStopRequested()
        batch = original_next_batch(validation=validation)
        if not validation:
            train_batches_since_launch += 1
        return batch

    trainer._next_batch = MethodType(controlled_next_batch, trainer)

    try:
        trainer.train()
    except StudioStopRequested:
        if stop_diagnostic_setting is not None:
            trainer.train_config.save_diagnostic_bundle = stop_diagnostic_setting
        print(
            f"Safe stop boundary reached: step={trainer.step:,} "
            f"tokens={trainer.tokens_seen:,}. Writing resumable checkpoint...",
            flush=True,
        )
        checkpoint = trainer._save(
            "studio-stop",
            permanent=False,
            tag="studio-stop",
        )
        trainer._flush_hub_uploads(close=True)
        print(
            f"Studio stop checkpoint saved: {checkpoint}",
            flush=True,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
