import torch

from asterlm import AsterConfig, AsterLM
from asterlm.generation import GenerationConfig, generate, generate_mtp_greedy


def test_mtp_reference_matches_greedy_output():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        max_seq_len=64,
        kda_ratio=1,
        kda_backend="torch",
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=2,
        mtp_rank=16,
        gradient_checkpointing=False,
    )
    model = AsterLM(config).eval()
    prompt = torch.randint(0, 64, (1, 8))
    standard = generate(model, prompt, GenerationConfig(max_new_tokens=6, temperature=0))
    speculative, _ = generate_mtp_greedy(model, prompt, 6)
    assert torch.equal(standard, speculative)
