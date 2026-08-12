#!/usr/bin/env python
from __future__ import annotations

import argparse

from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.generation.hub_checkpoint import resolve_hub_auto_resume
from asterlm.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain AsterLM")
    parser.add_argument("--model", default="configs/model/aster_220m.yaml")
    parser.add_argument("--train", default="configs/train/pretrain_laptop.yaml")
    parser.add_argument("--data", default="configs/data/pretrain_mixture_v1.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", default=None, help="Load weights only for a new training stage")
    group.add_argument("--resume", default=None, help="Resume model, optimizer, RNG, step, tokens, and data position")
    parser.add_argument("--hub-repo", default=None, help="Private Hugging Face model repo for milestone/final backups")
    parser.add_argument("--hub-public", action="store_true", help="Create/use a public Hub repo instead of private")
    parser.add_argument("--hub-model-only", action="store_true", help="Do not upload optimizer/RNG trainer_state.pt")
    parser.add_argument(
        "--run-class",
        choices=["exploratory", "decision_grade", "final"],
        default=None,
        help="Override the fail-closed training contract class",
    )
    parser.add_argument("--promotion-gates", default=None)
    args = parser.parse_args()
    train_config = TrainConfig.from_yaml(args.train)
    model_config = AsterConfig.from_yaml(args.model)
    if args.run_class:
        train_config.run_class = args.run_class
    if args.promotion_gates:
        train_config.promotion_gates_path = args.promotion_gates
    if args.resume:
        train_config.resume = args.resume
    if args.hub_repo:
        train_config.hub_repo_id = args.hub_repo
    if args.hub_public:
        train_config.hub_private = False
    if args.hub_model_only:
        train_config.hub_include_optimizer = False
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
    trainer = Trainer(
        model_config,
        train_config,
        DataConfig.from_yaml(args.data),
        mode="pretrain",
        initial_checkpoint=args.init_checkpoint,
    )
    print(trainer.model.architecture_summary())
    trainer.train()


if __name__ == "__main__":
    main()
