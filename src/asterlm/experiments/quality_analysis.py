from __future__ import annotations

import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from asterlm.experiments.quality import summarize_quality_run


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def normalized_curve_auc(curve: list[dict[str, Any]], x_key: str) -> float | None:
    points = sorted(
        (
            (float(row[x_key]), float(row["eval_main_loss"]))
            for row in curve
            if isinstance(row.get(x_key), (int, float))
            and isinstance(row.get("eval_main_loss"), (int, float))
        ),
        key=lambda item: item[0],
    )
    deduped: list[tuple[float, float]] = []
    for point in points:
        if deduped and point[0] == deduped[-1][0]:
            deduped[-1] = point
        else:
            deduped.append(point)
    if len(deduped) < 2 or deduped[-1][0] <= deduped[0][0]:
        return None
    area = sum(
        0.5 * (left[1] + right[1]) * (right[0] - left[0])
        for left, right in pairwise(deduped)
    )
    return area / (deduped[-1][0] - deduped[0][0])


def interpolate_loss(curve: list[dict[str, Any]], x_key: str, budget: float) -> float | None:
    points = sorted(
        (
            (float(row[x_key]), float(row["eval_main_loss"]))
            for row in curve
            if isinstance(row.get(x_key), (int, float))
            and isinstance(row.get("eval_main_loss"), (int, float))
        ),
        key=lambda item: item[0],
    )
    if not points or budget < points[0][0] or budget > points[-1][0]:
        return None
    for left, right in pairwise(points):
        if left[0] <= budget <= right[0]:
            if right[0] == left[0]:
                return right[1]
            fraction = (budget - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    return points[-1][1]


def first_budget_at_or_below(
    curve: list[dict[str, Any]],
    x_key: str,
    target_loss: float,
) -> float | None:
    """Interpolate the first budget where a learning curve reaches a loss target."""

    points = sorted(
        (
            (float(row[x_key]), float(row["eval_main_loss"]))
            for row in curve
            if isinstance(row.get(x_key), (int, float))
            and isinstance(row.get("eval_main_loss"), (int, float))
        ),
        key=lambda item: item[0],
    )
    deduped: list[tuple[float, float]] = []
    for point in points:
        if deduped and point[0] == deduped[-1][0]:
            deduped[-1] = point
        else:
            deduped.append(point)
    if not deduped:
        return None
    if deduped[0][1] <= target_loss:
        return deduped[0][0]
    for left, right in pairwise(deduped):
        if right[1] > target_loss:
            continue
        if right[1] == left[1]:
            return right[0]
        fraction = (left[1] - target_loss) / (left[1] - right[1])
        return left[0] + fraction * (right[0] - left[0])
    return None


def analyze_quality_campaign(campaign_path: str | Path) -> dict[str, Any]:
    campaign_path = Path(campaign_path).resolve()
    root = campaign_path.parent
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected = [
        (int(seed), str(run["candidate_id"]), str(run["execution_variant"]))
        for seed in campaign.get("seeds", [])
        for run in campaign.get("execution_matrix", [])
    ]
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed, candidate, variant in expected:
        run_dir = root / f"seed-{seed}" / candidate / variant
        summary = summarize_quality_run(run_dir)
        record = {
            "seed": seed,
            "candidate_id": candidate,
            "execution_variant": variant,
            "run_dir": run_dir.relative_to(root).as_posix(),
            **summary,
        }
        records.append(record)
        grouped[f"{candidate}:{variant}"].append(record)

    common_budgets: dict[int, dict[str, float]] = {}
    common_target_losses: dict[int, float] = {}
    for seed in campaign.get("seeds", []):
        seed_records = [row for row in records if row["seed"] == int(seed)]
        if len(seed_records) != len(campaign.get("execution_matrix", [])):
            continue
        budget: dict[str, float] = {}
        for key in ("wall_clock_total_seconds", "estimated_cumulative_flops"):
            maxima = [
                max(
                    float(point[key])
                    for point in row["learning_curve"]
                    if isinstance(point.get(key), (int, float))
                )
                for row in seed_records
                if any(isinstance(point.get(key), (int, float)) for point in row["learning_curve"])
            ]
            if len(maxima) == len(seed_records):
                budget[key] = min(maxima)
        common_budgets[int(seed)] = budget
        final_losses = [
            float(row["eval_main_loss"])
            for row in seed_records
            if row.get("status") == "ok"
            and isinstance(row.get("eval_main_loss"), (int, float))
        ]
        if len(final_losses) == len(seed_records):
            # The weakest terminal result is the strongest target every completed
            # arm is proven to reach. This is a non-extrapolated time-to-loss gate.
            common_target_losses[int(seed)] = max(final_losses)

    candidates: dict[str, Any] = {}
    for identity, items in grouped.items():
        complete = [item for item in items if item.get("status") == "ok"]
        losses = [float(item["eval_main_loss"]) for item in complete if item.get("eval_main_loss") is not None]
        throughput = [
            float(item["median_training_tokens_per_second"])
            for item in complete
            if item.get("median_training_tokens_per_second") is not None
        ]
        token_auc = [
            value
            for item in complete
            if (value := normalized_curve_auc(item["learning_curve"], "tokens_seen")) is not None
        ]
        wall_auc = [
            value
            for item in complete
            if (
                value := normalized_curve_auc(
                    item["learning_curve"], "wall_clock_total_seconds"
                )
            )
            is not None
        ]
        equal_wall = []
        equal_flops = []
        time_to_common_loss = []
        tokens_to_common_loss = []
        flops_to_common_loss = []
        for item in complete:
            budgets = common_budgets.get(int(item["seed"]), {})
            if "wall_clock_total_seconds" in budgets:
                value = interpolate_loss(
                    item["learning_curve"],
                    "wall_clock_total_seconds",
                    budgets["wall_clock_total_seconds"],
                )
                if value is not None:
                    equal_wall.append(value)
            if "estimated_cumulative_flops" in budgets:
                value = interpolate_loss(
                    item["learning_curve"],
                    "estimated_cumulative_flops",
                    budgets["estimated_cumulative_flops"],
                )
                if value is not None:
                    equal_flops.append(value)
            target = common_target_losses.get(int(item["seed"]))
            if target is not None:
                for x_key, destination in (
                    ("wall_clock_total_seconds", time_to_common_loss),
                    ("tokens_seen", tokens_to_common_loss),
                    ("estimated_cumulative_flops", flops_to_common_loss),
                ):
                    value = first_budget_at_or_below(
                        item["learning_curve"], x_key, target
                    )
                    if value is not None:
                        destination.append(value)
        candidates[identity] = {
            "candidate_id": items[0]["candidate_id"],
            "execution_variant": items[0]["execution_variant"],
            "complete_seed_count": len(complete),
            "expected_seed_count": len(items),
            "final_eval_loss_mean": _mean(losses),
            "final_eval_loss_stdev": _stdev(losses),
            "token_curve_auc_mean": _mean(token_auc),
            "wall_curve_auc_mean": _mean(wall_auc),
            "equal_wall_loss_mean": _mean(equal_wall),
            "equal_active_flops_loss_mean": _mean(equal_flops),
            "time_to_common_loss_seconds_mean": _mean(time_to_common_loss),
            "tokens_to_common_loss_mean": _mean(tokens_to_common_loss),
            "active_flops_to_common_loss_mean": _mean(flops_to_common_loss),
            "median_training_tokens_per_second": statistics.median(throughput) if throughput else None,
            "mean_gpu_util_percent": _mean(
                [float(item["mean_gpu_util_percent"]) for item in complete if item.get("mean_gpu_util_percent") is not None]
            ),
            "peak_vram_gib": max(
                [float(item["peak_vram_gib"]) for item in complete if item.get("peak_vram_gib") is not None],
                default=None,
            ),
            "run_survival_rate": _mean(
                [float(bool(item.get("run_survived"))) for item in items]
            ),
            "training_loss_nonfinite_count": sum(
                int(item.get("training_loss_nonfinite_count") or 0) for item in items
            ),
            "gradient_nonfinite_count": sum(
                int(item.get("gradient_nonfinite_count") or 0) for item in items
            ),
            "gradient_norm_p95_mean": _mean(
                [
                    float(item["gradient_norm_p95"])
                    for item in complete
                    if item.get("gradient_norm_p95") is not None
                ]
            ),
            "gradient_norm_max": max(
                [
                    float(item["gradient_norm_max"])
                    for item in complete
                    if item.get("gradient_norm_max") is not None
                ],
                default=None,
            ),
            "gradient_clip_fraction_mean": _mean(
                [
                    float(item["gradient_clip_fraction"])
                    for item in complete
                    if item.get("gradient_clip_fraction") is not None
                ]
            ),
            "loss_upward_jump_gt_0_5_count": sum(
                int(item.get("loss_upward_jump_gt_0_5_count") or 0) for item in items
            ),
            "parameter_global_rms_relative_drift_mean": _mean(
                [
                    float(item["parameter_global_rms_relative_drift"])
                    for item in complete
                    if item.get("parameter_global_rms_relative_drift") is not None
                ]
            ),
            "optimizer_wall_fraction_mean": _mean(
                [
                    float(item["optimizer_wall_fraction_mean"])
                    for item in complete
                    if item.get("optimizer_wall_fraction_mean") is not None
                ]
            ),
            "muon_relative_update_rms_mean": _mean(
                [
                    float(item["muon_relative_update_rms_mean"])
                    for item in complete
                    if item.get("muon_relative_update_rms_mean") is not None
                ]
            ),
        }
    expected_runs = len(expected)
    complete_runs = sum(record.get("status") == "ok" for record in records)
    return {
        "schema_version": 1,
        "campaign": campaign_path.as_posix(),
        "campaign_status": campaign.get("status"),
        "analysis_status": "complete" if complete_runs == expected_runs else "partial",
        "expected_runs": expected_runs,
        "complete_runs": complete_runs,
        "common_budgets_by_seed": common_budgets,
        "common_target_loss_by_seed": common_target_losses,
        "candidates": candidates,
        "runs": records,
        "selection": {
            "status": "not_selected",
            "reason": (
                "Campaign analysis is incomplete"
                if complete_runs < expected_runs
                else (
                    "Optimizer choice still requires replicated longer-horizon time-to-quality and recovery gates"
                    if campaign.get("campaign_type") == "optimizer_quality"
                    else "Architecture mechanism still requires long-context and native-scale promotion gates"
                )
            ),
        },
    }
