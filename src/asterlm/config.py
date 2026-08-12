from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


@dataclass(slots=True)
class AsterConfig:
    # Token/model dimensions
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    head_dim: int = 64
    ffn_hidden: int = 2048
    ffn_type: str = "dense"  # dense | moe | latent_moe
    moe_every: int = 1
    moe_first_dense_layers: int = 0
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_shared_experts: int = 1
    moe_expert_hidden: int = 512
    # LatentMoE compresses only the routed branch; router/shared experts stay full-width.
    latent_moe_dim: int | None = None
    # Kimi-K3-inspired Stable LatentMoE ablation: RMS-normalize the aggregated routed latent before up-projection.
    latent_moe_post_norm: bool = False
    # Kimi K3 Stable LatentMoE uses bounded SiTU-GLU in both routed and shared experts.
    moe_activation: str = "swiglu"  # swiglu | situ_glu
    moe_situ_beta_gate: float = 4.0
    moe_situ_beta_up: float = 25.0
    moe_aux_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001
    moe_router_score: str = "sigmoid"  # sigmoid (DeepSeek-style) | softmax
    moe_balance_strategy: str = "bias"  # bias | aux_loss | hybrid | quantile
    moe_router_bias_update_speed: float = 0.001
    moe_quantile_bins: int = 256
    moe_quantile_margin_bound: float = 4.0
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    # Linear implementation. Transformer Engine is optional and is only useful on
    # CUDA GPUs with working FP8 kernels; checkpoints remain shape-compatible with
    # the ordinary torch implementation.
    linear_backend: str = "torch"  # torch | transformer_engine
    # FFN/expert-only backend. `loqt_int4` stores the largest matrices in INT4 and
    # trains low-rank updates that are periodically merged. Attention stays BF16/FP8.
    ffn_linear_backend: str | None = None  # None | torch | transformer_engine | loqt_int4
    loqt_rank: int = 32
    loqt_alpha: float = 32.0
    loqt_group_size: int = 64

    # Hybrid mixer. One latent-attention layer follows each kda_ratio KDA layers.
    kda_ratio: int = 3
    layer_pattern: list[str] | None = None
    kda_backend: str = "auto"  # auto | fla | torch
    # KDA projection/state geometry is independent from MLA geometry. Published
    # Kimi Linear uses 128-wide KDA heads even though its outer model/MLA head
    # geometry differs; leaving these unset preserves the compact legacy shape.
    kda_num_heads: int | None = None
    kda_head_dim: int | None = None
    kda_expand_v: float = 1.0
    kda_short_conv: bool = True
    kda_conv_size: int = 4
    kda_safe_gate: bool = True
    kda_lower_bound: float | None = -5.0
    kda_allow_negative_eigenvalues: bool = False

    # Memory-compressed latent attention.
    latent_rank: int = 64
    q_lora_rank: int | None = None
    latent_rms_norm: bool = True
    rope_dim: int = 32
    attention_window: int | None = 4096
    sink_tokens: int = 64
    attention_dropout: float = 0.0
    attention_gate: bool = True
    logit_softcap: float | None = None
    qk_stat_tokens: int = 32

    # Training-time global-attention backend. Inference keeps the exact latent-cache path.
    attention_train_backend: str = "sdpa"  # sdpa | absorbed_sdpa | flex_window
    attention_train_window: int = 4096
    attention_train_global_stride: int = 0
    attention_flex_block_size: int = 128

    # DeepSeek-V4-style compressed attention research geometry. ``csa`` and
    # ``hca`` are explicit layer_pattern kinds so they cannot silently replace
    # MLA. The portable implementation is a training/reference oracle until an
    # optimized cached backend passes parity.
    compressed_attention_local_window: int = 128
    compressed_attention_hca_ratio: int = 128
    compressed_attention_index_topk: int = 1024
    compressed_attention_index_n_heads: int = 64
    compressed_attention_index_head_dim: int = 128
    compressed_attention_output_groups: int = 1
    compressed_attention_output_lora_rank: int | None = None
    compressed_attention_rope_theta: float = 160_000.0
    compressed_attention_backend: str = "reference"  # reference | triton | auto

    # Inference cache storage. KDA layers use fixed recurrent states; these options
    # apply to the latent/global-attention layers only. `hadamard_int4` is a
    # TurboQuant-inspired, training-free rotated INT4 reference path. It saves VRAM
    # but is not claimed to match Google's fused TurboQuant kernels.
    cache_dtype: str = "bfloat16"  # bfloat16 | float8 | int8 | int4 | hadamard_int4
    cache_group_size: int = 64
    cache_recent_tokens: int = 512
    cache_chunk_tokens: int = 1024
    cache_quantize_rope: bool = False

    # Position encoding.
    rope_theta: float = 1_000_000.0
    rope_scaling_type: str = "none"  # none | linear | yarn
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # Residual/FFN.
    rms_eps: float = 1e-6
    norm_type: str = "rmsnorm"  # rmsnorm | ssnorm (Outlier-Safe Pre-Training)
    embedding_projection: bool = False
    residual_dropout: float = 0.0
    ffn_dropout: float = 0.0
    use_block_attnres: bool = False
    attnres_block_size: int = 4
    attnres_key_dim: int = 64
    residual_architecture: str = "standard"  # standard | mhc
    mhc_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_eps: float = 1e-6

    # Multi-token prediction; heads predict t+2, t+3, ... beyond the main next-token head.
    mtp_depth: int = 2
    mtp_rank: int = 256
    mtp_loss_weight: float = 0.15
    # MTP implementations are ablated independently; DeepSeek-style uses one
    # sequential future-token module with a full Aster block.
    mtp_architecture: str = "low_rank"  # low_rank | deepseek
    mtp_block_kind: str = "latent"  # latent | kda | gdn2
    lm_loss_chunk_size: int = 256
    # PyTorch 2.13 LinearCrossEntropy avoids materializing [B,T,V] logits.
    lm_loss_backend: str = "legacy_chunked"  # legacy_chunked | torch_linear_ce
    linear_ce_chunking_method: str = "auto"
    linear_ce_acc_policy: str = "compact"

    # Stability and execution.
    qk_clip_tau: float = 100.0
    init_std: float = 0.02
    residual_init_scale: float | None = None
    gradient_checkpointing: bool = True
    # Group multiple transformer blocks behind one checkpoint boundary at long context.
    checkpoint_segment_size: int = 1

    def __post_init__(self) -> None:
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError("rope_dim must be a positive even number")
        if self.ffn_hidden <= self.d_model:
            raise ValueError("ffn_hidden should exceed d_model")
        if self.ffn_type not in {"dense", "moe", "latent_moe"}:
            raise ValueError("ffn_type must be dense, moe, or latent_moe")
        if self.ffn_type == "latent_moe":
            latent_dim = self.latent_moe_dim or (self.d_model // 4)
            if not 0 < latent_dim < self.d_model:
                raise ValueError("latent_moe_dim must be in (0, d_model)")
        if self.linear_backend not in {"torch", "transformer_engine"}:
            raise ValueError("linear_backend must be torch or transformer_engine")
        if self.ffn_linear_backend not in {None, "torch", "transformer_engine", "loqt_int4"}:
            raise ValueError("unsupported ffn_linear_backend")
        if self.loqt_rank <= 0 or self.loqt_alpha <= 0 or self.loqt_group_size <= 0:
            raise ValueError("LoQT rank, alpha, and group size must be positive")
        if self.moe_every <= 0 or self.moe_first_dense_layers < 0:
            raise ValueError("invalid MoE placement settings")
        if self.moe_num_experts <= 0 or not 1 <= self.moe_top_k <= self.moe_num_experts:
            raise ValueError("invalid MoE expert/top-k settings")
        if self.moe_shared_experts < 0 or self.moe_expert_hidden <= 0:
            raise ValueError("invalid shared-expert dimensions")
        if self.moe_aux_loss_weight < 0 or self.moe_router_z_loss_weight < 0:
            raise ValueError("MoE loss weights must be non-negative")
        if self.moe_router_score not in {"sigmoid", "softmax"}:
            raise ValueError("moe_router_score must be sigmoid or softmax")
        if self.moe_activation not in {"swiglu", "situ_glu"}:
            raise ValueError("moe_activation must be swiglu or situ_glu")
        if self.moe_situ_beta_gate <= 0 or self.moe_situ_beta_up <= 0:
            raise ValueError("SiTU-GLU beta values must be positive")
        if self.moe_balance_strategy not in {"bias", "aux_loss", "hybrid", "quantile"}:
            raise ValueError(
                "moe_balance_strategy must be bias, aux_loss, hybrid, or quantile"
            )
        if self.moe_router_bias_update_speed < 0:
            raise ValueError("moe_router_bias_update_speed must be non-negative")
        if self.moe_quantile_bins < 16 or self.moe_quantile_margin_bound <= 0:
            raise ValueError("quantile balancing requires >=16 bins and a positive margin bound")
        if self.kda_ratio < 0:
            raise ValueError("kda_ratio must be non-negative")
        if self.layer_pattern is not None:
            bad = set(self.layer_pattern) - {"kda", "gdn2", "latent", "csa", "hca"}
            if bad:
                raise ValueError(f"Unsupported layer kinds: {sorted(bad)}")
            if len(self.layer_pattern) != self.n_layers:
                raise ValueError("layer_pattern length must equal n_layers")
        if self.kda_backend not in {"auto", "fla", "torch"}:
            raise ValueError("kda_backend must be auto, fla, or torch")
        if self.kda_num_heads is not None and self.kda_num_heads <= 0:
            raise ValueError("kda_num_heads must be positive when set")
        if self.kda_head_dim is not None and self.kda_head_dim <= 0:
            raise ValueError("kda_head_dim must be positive when set")
        if self.kda_expand_v <= 0:
            raise ValueError("kda_expand_v must be positive")
        if self.rope_scaling_type not in {"none", "linear", "yarn"}:
            raise ValueError("rope_scaling_type must be none, linear, or yarn")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.attention_train_backend not in {"sdpa", "absorbed_sdpa", "flex_window"}:
            raise ValueError("attention_train_backend must be sdpa, absorbed_sdpa, or flex_window")
        if self.attention_train_window <= 0 or self.attention_flex_block_size <= 0:
            raise ValueError("training attention window/block size must be positive")
        if self.attention_train_global_stride < 0:
            raise ValueError("attention_train_global_stride must be non-negative")
        if self.attention_train_backend == "flex_window" and self.attention_dropout != 0.0:
            raise ValueError("vNext flex_window currently requires attention_dropout=0")
        if self.compressed_attention_local_window <= 0:
            raise ValueError("compressed attention local window must be positive")
        if self.compressed_attention_hca_ratio <= 1:
            raise ValueError("compressed HCA ratio must exceed one")
        if self.compressed_attention_rope_theta <= 0:
            raise ValueError("compressed attention RoPE theta must be positive")
        if self.compressed_attention_backend not in {"reference", "triton", "auto"}:
            raise ValueError("compressed attention backend must be reference, triton, or auto")
        if self.compressed_attention_index_topk <= 0:
            raise ValueError("compressed attention index top-k must be positive")
        if (
            self.compressed_attention_index_n_heads <= 0
            or self.compressed_attention_index_head_dim < self.rope_dim
        ):
            raise ValueError("compressed attention index geometry is invalid")
        if self.n_heads % self.compressed_attention_output_groups:
            raise ValueError("compressed attention output groups must divide n_heads")
        if (
            self.compressed_attention_output_lora_rank is not None
            and self.compressed_attention_output_lora_rank <= 0
        ):
            raise ValueError("compressed attention output LoRA rank must be positive")
        if self.norm_type not in {"rmsnorm", "ssnorm"}:
            raise ValueError("norm_type must be rmsnorm or ssnorm")
        if not 0.0 <= self.residual_dropout < 1.0 or not 0.0 <= self.ffn_dropout < 1.0:
            raise ValueError("residual and FFN dropout must be in [0, 1)")
        if self.residual_architecture not in {"standard", "mhc"}:
            raise ValueError("residual_architecture must be standard or mhc")
        if self.mhc_streams <= 0 or self.mhc_sinkhorn_iterations <= 0 or self.mhc_eps <= 0:
            raise ValueError("mHC streams, iterations, and epsilon must be positive")
        if self.residual_architecture == "mhc" and self.use_block_attnres:
            raise ValueError("mHC and Block AttnRes are mutually exclusive research candidates")
        if self.residual_architecture == "mhc" and self.mtp_depth > 0:
            raise ValueError("mHC MTP integration is not yet reference-parity validated")
        if self.latent_rank <= 0:
            raise ValueError("latent_rank must be positive")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive when enabled")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if self.attention_window is not None and self.attention_window <= self.sink_tokens:
            raise ValueError("attention_window must exceed sink_tokens")
        if self.logit_softcap is not None and self.logit_softcap <= 0:
            raise ValueError("logit_softcap must be positive when enabled")
        if self.qk_clip_tau <= 0:
            raise ValueError("qk_clip_tau must be positive")
        if self.cache_dtype not in {"bfloat16", "float8", "int8", "int4", "hadamard_int4"}:
            raise ValueError("unsupported cache_dtype")
        if self.cache_group_size <= 0 or self.cache_chunk_tokens <= 0:
            raise ValueError("cache group/chunk sizes must be positive")
        if self.cache_recent_tokens < 0:
            raise ValueError("cache_recent_tokens must be non-negative")
        if self.mtp_depth < 0 or self.mtp_rank <= 0 or self.mtp_loss_weight < 0:
            raise ValueError("MTP depth/weight must be non-negative and rank positive")
        if self.mtp_architecture not in {"low_rank", "deepseek"}:
            raise ValueError("mtp_architecture must be low_rank or deepseek")
        if self.mtp_block_kind not in {"latent", "kda", "gdn2"}:
            raise ValueError("mtp_block_kind must be latent, kda, or gdn2")
        if self.mtp_architecture == "deepseek" and self.mtp_depth not in {0, 1}:
            raise ValueError("Aster vNext2 validates DeepSeek-style MTP at depth 1 only")
        if self.lm_loss_chunk_size <= 0:
            raise ValueError("lm_loss_chunk_size must be positive")
        if self.lm_loss_backend not in {"legacy_chunked", "torch_linear_ce"}:
            raise ValueError("lm_loss_backend must be legacy_chunked or torch_linear_ce")
        if self.linear_ce_acc_policy not in {"auto", "compact", "accurate"}:
            raise ValueError("linear_ce_acc_policy must be auto, compact, or accurate")
        if self.max_seq_len <= 0 or self.vocab_size <= 0 or self.n_layers <= 0:
            raise ValueError("model sizes must be positive")
        if self.checkpoint_segment_size <= 0:
            raise ValueError("checkpoint_segment_size must be positive")

    @property
    def ffn_backend(self) -> str:
        return self.linear_backend if self.ffn_linear_backend is None else self.ffn_linear_backend

    @property
    def pattern(self) -> list[str]:
        if self.layer_pattern is not None:
            return list(self.layer_pattern)
        if self.kda_ratio == 0:
            return ["latent"] * self.n_layers
        cycle = ["kda"] * self.kda_ratio + ["latent"]
        return [cycle[i % len(cycle)] for i in range(self.n_layers)]

    @classmethod
    def from_yaml(cls, path: str | Path) -> AsterConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "model" in values:
            values = values["model"]
        return _construct(cls, values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "runs/aster"
    seed: int = 1337
    # Exploratory runs remain easy to launch. Decision-grade and final runs are
    # fail-closed by the executable training contract in training/contracts.py.
    run_class: str = "exploratory"  # exploratory | decision_grade | final
    promotion_gates_path: str = "configs/experiments/promotion_gates.yaml"
    # Architecture campaigns can initialize each ordinary projection from its
    # qualified module name. This prevents a changed KDA/MoE tensor shape from
    # shifting the RNG stream used by otherwise-identical later layers.
    deterministic_named_initialization: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"
    matmul_precision: str = "high"
    compile: bool = False
    compile_mode: str = "default"
    # Runtime selection is explicit and audited. External engines are campaign
    # candidates until an adapter passes parity, recovery, and throughput gates.
    execution_backend: str = "auto"  # auto | aster_local | megatron_core | torchtitan | deepspeed
    execution_autotune: bool = True
    # Physical expert execution only; this never changes routing/model semantics.
    moe_implementation: str = "auto"  # auto | reference | grouped | cutlass | torch_grouped | liger
    cuda_graphs: bool = False
    precision_backend: str = "amp"  # amp | transformer_engine_fp8
    fp8_format: str = "hybrid"  # hybrid | e4m3
    fp8_recipe: str = "delayed"  # delayed | current
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"
    activation_offload: bool = False
    activation_offload_pin_memory: bool = True
    loqt_merge_interval: int = 0
    loqt_merge_on_cpu: bool = True
    loqt_reset_optimizer_state: bool = True

    sequence_length: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = 100_000
    max_tokens: int | None = None

    # `adamw` is the mandatory dense/control optimizer. Other methods must beat it
    # rather than borrowing an AdamW result from a differently partitioned hybrid.
    optimizer: str = "muon_adamw"  # adamw | muon_adamw* | apollo* | torchao_adamw*
    muon_lr: float = 0.01
    adam_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2_000
    schedule_type: str = "wsd"  # wsd | cosine | constant
    decay_fraction: float = 0.1
    decay_shape: str = "cosine"  # cosine | linear | sqrt
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    muon_update_rms: float = 0.2
    # Kimi K3 orthogonalizes Q/K/V momentum independently per attention head.
    muon_per_head: bool = False
    # Batch equal-shaped matrices through one orthogonalization call. The workspace
    # bound prevents routed-expert mega-batches from consuming all spare VRAM.
    muon_megabatch: bool = True
    muon_megabatch_max_gib: float = 0.5
    # GPU-resident Muon momentum. INT8 uses blockwise abs-max quantization; it is
    # optimizer-state compression, never CPU/NVMe offload.
    muon_state_dtype: str = "float32"  # float32 | int8_blockwise
    muon_quant_block_size: int = 2048
    max_grad_norm: float = 1.0

    # APOLLO/APOLLO-Mini: low-rank optimizer states for VRAM-constrained full pretraining.
    apollo_rank: int = 1
    apollo_scale: float = 128.0
    apollo_scale_type: str = "tensor"  # tensor (Mini) | channel
    apollo_update_proj_gap: int = 200
    apollo_proj: str = "random"
    apollo_proj_type: str = "std"
    apollo_scale_front: bool = False
    apollo_disable_norm_limiter: bool = False

    qk_clip_interval: int = 100
    log_interval: int = 10
    eval_interval: int = 1_000
    eval_batches: int = 32
    save_interval: int = 1_000
    # Wall-clock safety trigger for machines/providers whose step durations differ.
    # A periodic checkpoint fires when either this timer or save_interval is due.
    checkpoint_interval_minutes: float | None = None
    keep_last_checkpoints: int = 3
    # Exploratory architecture campaigns can be metrics-only to avoid retaining
    # multi-gigabyte model/optimizer states. `final_only` keeps one terminal
    # recovery point; `full` enables periodic, milestone, and final checkpoints.
    checkpoint_policy: str = "full"  # none | final_only | full
    # Retain one additional full-state checkpoint in exponentially older bands.
    # This keeps dense recent recovery points plus a logarithmic historical spine.
    checkpoint_pyramid_levels: int = 0
    # Hard ceiling for complete local checkpoint directories. Research milestones
    # may leave the local cache only after their full-state Hub copy is hash-verified.
    # Set to null to disable the byte budget (not permitted for final runs).
    checkpoint_local_budget_gib: float | None = None
    # Permanent, token-addressed checkpoints for scaling/grokking analysis. These
    # survive ordinary rolling-checkpoint retention and can be uploaded to the Hub.
    milestone_tokens: list[int] = field(default_factory=list)
    milestone_eval: bool = True
    resume: str | None = None

    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int | None = None
    shuffle_buffer: int = 10_000
    tokenizer_path: str = "artifacts/tokenizer.json"
    # Protected final runs bind the tokenizer to the exact clean data config.
    tokenizer_manifest_path: str | None = None
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    ignore_index: int = -100

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    # Stable physical-job identity. The trainer creates and persists this before
    # first contact with W&B, then requires the same run on exact resume.
    wandb_run_id: str | None = None
    tensorboard: bool = False
    jsonl_metrics: bool = True
    system_metrics_interval: float = 5.0
    diagnostic_interval: int = 100
    save_diagnostic_bundle: bool = True

    # Optional resilient Hugging Face model-repository synchronization. The repo ID
    # can also be supplied at runtime through ASTERLM_HUB_REPO_ID.
    hub_repo_id: str | None = None
    hub_private: bool = True
    hub_revision: str = "main"
    hub_upload_every_save: bool = False
    hub_upload_milestones: bool = True
    hub_upload_final: bool = True
    hub_upload_on_stop: bool = True
    hub_auto_resume_latest: bool = False
    hub_include_optimizer: bool = True
    hub_fail_on_error: bool = False
    # Decimal TB, matching provider quota displays. The lower operational guard
    # is a fail-closed review point; the hard cap may never be exceeded.
    hub_storage_guard_tb_decimal: float | None = None
    hub_storage_hard_cap_tb_decimal: float | None = None

    def __post_init__(self) -> None:
        if self.run_class not in {"exploratory", "decision_grade", "final"}:
            raise ValueError("run_class must be exploratory, decision_grade, or final")
        if self.execution_backend not in {
            "auto",
            "aster_local",
            "megatron_core",
            "torchtitan",
            "deepspeed",
        }:
            raise ValueError("unsupported execution_backend")
        if self.moe_implementation not in {
            "auto",
            "reference",
            "grouped",
            "cutlass",
            "torch_grouped",
            "liger",
        }:
            raise ValueError("unsupported moe_implementation")
        if self.optimizer not in {
            "adamw",
            "muon_adamw",
            "muon_adamw8bit",
            "apollo_mini",
            "apollo",
            "torchao_adamw8bit",
            "torchao_adamw4bit",
            "torchao_cpu_offload_adamw",
        }:
            raise ValueError("unsupported optimizer")
        if self.precision_backend not in {"amp", "transformer_engine_fp8"}:
            raise ValueError("precision_backend must be amp or transformer_engine_fp8")
        if self.fp8_format not in {"hybrid", "e4m3"}:
            raise ValueError("fp8_format must be hybrid or e4m3")
        if self.fp8_recipe not in {"delayed", "current"}:
            raise ValueError("fp8_recipe must be delayed or current")
        if self.fp8_amax_history_len <= 0:
            raise ValueError("fp8_amax_history_len must be positive")
        if self.loqt_merge_interval < 0:
            raise ValueError("loqt_merge_interval must be non-negative")
        if self.system_metrics_interval <= 0 or self.diagnostic_interval <= 0:
            raise ValueError("telemetry intervals must be positive")
        if self.schedule_type not in {"wsd", "cosine", "constant"}:
            raise ValueError("schedule_type must be wsd, cosine, or constant")
        if self.decay_shape not in {"cosine", "linear", "sqrt"}:
            raise ValueError("decay_shape must be cosine, linear, or sqrt")
        if not 0.0 < self.decay_fraction <= 1.0:
            raise ValueError("decay_fraction must be in (0, 1]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if self.sequence_length <= 0 or self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch and sequence dimensions must be positive")
        if self.max_steps <= 0 or (self.max_tokens is not None and self.max_tokens <= 0):
            raise ValueError("max_steps and max_tokens must be positive")
        if any(token <= 0 for token in self.milestone_tokens):
            raise ValueError("milestone_tokens must contain only positive integers")
        if self.keep_last_checkpoints < 0 or self.checkpoint_pyramid_levels < 0:
            raise ValueError("checkpoint retention values must be non-negative")
        if self.checkpoint_interval_minutes is not None and self.checkpoint_interval_minutes <= 0:
            raise ValueError("checkpoint_interval_minutes must be positive when configured")
        if self.checkpoint_local_budget_gib is not None and self.checkpoint_local_budget_gib <= 0:
            raise ValueError("checkpoint_local_budget_gib must be positive when configured")
        if self.checkpoint_policy not in {"none", "final_only", "full"}:
            raise ValueError("checkpoint_policy must be none, final_only, or full")
        if self.milestone_tokens != sorted(set(self.milestone_tokens)):
            raise ValueError("milestone_tokens must be sorted and unique")
        if self.max_tokens is not None and any(token > self.max_tokens for token in self.milestone_tokens):
            raise ValueError("milestone_tokens cannot exceed max_tokens")
        if not self.hub_revision.strip():
            raise ValueError("hub_revision cannot be empty")
        for name, value in (
            ("hub_storage_guard_tb_decimal", self.hub_storage_guard_tb_decimal),
            ("hub_storage_hard_cap_tb_decimal", self.hub_storage_hard_cap_tb_decimal),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")
        if (
            self.hub_storage_guard_tb_decimal is not None
            and self.hub_storage_hard_cap_tb_decimal is not None
            and self.hub_storage_guard_tb_decimal > self.hub_storage_hard_cap_tb_decimal
        ):
            raise ValueError("Hub storage guard cannot exceed the hard cap")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.muon_lr <= 0 or self.adam_lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning rates and max_grad_norm must be positive")
        if self.muon_update_rms <= 0 or self.muon_ns_steps <= 0:
            raise ValueError("Muon update RMS and Newton-Schulz steps must be positive")
        if self.muon_megabatch_max_gib <= 0:
            raise ValueError("Muon mega-batch workspace must be positive")
        if self.muon_state_dtype not in {"float32", "int8_blockwise"}:
            raise ValueError("muon_state_dtype must be float32 or int8_blockwise")
        if self.muon_quant_block_size <= 0:
            raise ValueError("muon_quant_block_size must be positive")
        if self.apollo_rank <= 0 or self.apollo_scale <= 0 or self.apollo_update_proj_gap <= 0:
            raise ValueError("APOLLO rank, scale, and update gap must be positive")
        if self.apollo_scale_type not in {"tensor", "channel"}:
            raise ValueError("apollo_scale_type must be tensor or channel")
        if self.optimizer == "apollo_mini" and (self.apollo_rank != 1 or self.apollo_scale_type != "tensor"):
            raise ValueError("apollo_mini requires apollo_rank=1 and apollo_scale_type=tensor")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "train" in values:
            values = values["train"]
        if "adam_betas" in values:
            values["adam_betas"] = tuple(values["adam_betas"])
        return _construct(cls, values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceConfig:
    path: str
    weight: float
    name: str | None = None
    split: str = "train"
    text_field: str = "text"
    revision: str | None = None
    streaming: bool = True
    trust_remote_code: bool = False
    data_files: str | list[str] | None = None
    format: str = "text"  # text | messages | prompt_response | chosen_rejected
    prompt_field: str = "prompt"
    response_field: str = "response"
    messages_field: str = "messages"
    fim_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Source weight must be positive")
        if not 0.0 <= self.fim_rate <= 1.0:
            raise ValueError("fim_rate must be between 0 and 1")
        if self.format not in {"text", "messages", "prompt_response", "chosen_rejected"}:
            raise ValueError("Unsupported source format")


@dataclass(slots=True)
class DataConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    validation_sources: list[SourceConfig] = field(default_factory=list)
    seed: int = 1337
    shuffle_buffer: int = 10_000
    min_chars: int = 64
    max_chars: int = 200_000
    quality_filters: bool = True
    add_eos_between_documents: bool = True
    mask_cross_document_loss: bool = True
    # Optional provenance sidecar for exact corpus quotas, hashes, and split rules.
    manifest_path: str | None = None
    validation_role: str = "pipeline_smoke"

    def __post_init__(self) -> None:
        if self.validation_role not in {
            "pipeline_smoke",
            "optimization_validation",
            "architecture_holdout",
            "benchmark_holdout",
            "long_context_holdout",
        }:
            raise ValueError("unsupported validation_role")

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "data" in values:
            values = values["data"]
        src = [_construct(SourceConfig, item) for item in values.pop("sources", [])]
        val = [_construct(SourceConfig, item) for item in values.pop("validation_sources", [])]
        return cls(sources=src, validation_sources=val, **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
