#!/usr/bin/env python
from __future__ import annotations

import argparse

from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain AsterLM")
    parser.add_argument("--model", default="configs/model/aster_220m.yaml")
    parser.add_argument("--train", default="configs/train/pretrain_laptop.yaml")
    parser.add_argument("--data", default="configs/data/pretrain_mixture_v1.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", default=None, help="Load weights only for a new training stage")
    group.add_argument("--resume", default=None, help="Resume model, optimizer, RNG, step, tokens, and data position")
    args = parser.parse_args()
    train_config = TrainConfig.from_yaml(args.train)
    if args.resume:
        train_config.resume = args.resume
    trainer = Trainer(
        AsterConfig.from_yaml(args.model),
        train_config,
        DataConfig.from_yaml(args.data),
        mode="pretrain",
        initial_checkpoint=args.init_checkpoint,
    )
    print(trainer.model.architecture_summary())
    trainer.train()


if __name__ == "__main__":
    main()
