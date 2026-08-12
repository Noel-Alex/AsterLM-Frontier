from __future__ import annotations

from typing import Any

from asterlm.config import TrainConfig


def build_analysis_manifest(config: TrainConfig) -> dict[str, Any]:
    """Describe the durable analysis surface and its collection cadence."""

    return {
        "schema_version": 1,
        "axis": {"primary": "tokens_seen", "secondary": ["step", "wall_time_unix"]},
        "cadence": {
            "training_window_steps": config.log_interval,
            "system_min_seconds": config.system_metrics_interval,
            "deep_diagnostics_steps": config.diagnostic_interval,
            "evaluation_steps": config.eval_interval,
            "checkpoint_steps": config.save_interval,
            "token_milestones": config.milestone_tokens,
        },
        "metric_groups": {
            "objective": ["loss", "main_loss", "mtp_loss", "router_aux_loss", "router_z_loss"],
            "validation": ["eval_loss", "eval_main_loss", "eval_perplexity", "eval_role"],
            "progress": [
                "tokens_seen", "progress_fraction", "tokens_per_logical_parameter",
                "tokens_per_active_parameter", "wall_clock_campaign_seconds",
                "eta_seconds", "eta_smoothed_seconds",
            ],
            "throughput": [
                "tokens_per_second", "tokens_per_second_ema", "estimated_training_tflops",
                "estimated_cumulative_flops", "effective_batch_tokens",
            ],
            "phase_timing": [
                "data_wait_seconds", "forward_submit_seconds", "backward_submit_seconds",
                "optimizer_submit_seconds", "window_seconds", "window_excluded_seconds",
            ],
            "optimization": [
                "lr_multiplier", "grad_norm_pre_clip", "grad_clip_coefficient",
                "grad_was_clipped", "grad_global_l2_unclipped", "grad_max_abs",
                "grad_all_finite", "muon_update_rms", "adamw_update_rms",
            ],
            "parameters": ["param_global_l2", "param_global_rms", "param_rms_*", "param_max_abs_*"],
            "moe": ["moe_*", "router_*", "expert_*"],
            "data_cursor": ["data_source_epoch_*", "data_file_index_*", "data_record_index_*"],
            "attention": ["qk_heads_clipped", "qk_max_logit", "kda_*", "attention_*"],
            "system": [
                "cuda_*", "gpu_*", "host_*", "gpu_energy_joules_total",
                "gpu_energy_kwh_total",
            ],
            "durability": [
                "checkpoint_*", "hub_sync_ok", "hub_sync_seconds", "hub_sync_error",
            ],
        },
        "durable_artifacts": {
            "local": [
                "run_manifest.json", "experiment.json", "metrics.jsonl",
                "analysis_manifest.json", "tensorboard/", "diagnostics/",
                "hub-verifications/", "checkpoint-*/checkpoint_manifest.json",
            ],
            "wandb": "scalar history plus checkpoint/failure diagnostic artifacts",
            "huggingface": "private hash-verified full-state checkpoints and run ledger",
        },
        "notes": [
            "Energy is observational only and never ranks training decisions.",
            "Wildcard metric names are schema families; concrete keys remain append-only in metrics.jsonl.",
            "Heavy diagnostics are periodic to preserve training throughput.",
        ],
    }
