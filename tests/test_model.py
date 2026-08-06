import torch

from asterlm import AsterConfig, AsterLM


def tiny_config(**changes):
    values = dict(
        vocab_size=96,
        d_model=64,
        n_layers=4,
        n_heads=4,
        head_dim=16,
        ffn_hidden=176,
        max_seq_len=64,
        kda_ratio=1,
        kda_backend="torch",
        latent_rank=16,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=2,
        mtp_rank=32,
        gradient_checkpointing=False,
    )
    values.update(changes)
    return AsterConfig(**values)


def test_forward_backward_and_mtp():
    model = AsterLM(tiny_config())
    ids = torch.randint(0, 96, (2, 16))
    labels = torch.randint(0, 96, (2, 16))
    output = model(ids, labels=labels, return_mtp=True)
    assert output.logits.shape == (2, 16, 96)
    assert len(output.mtp_logits) == 2
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cached_decode_matches_full_forward():
    torch.manual_seed(4)
    model = AsterLM(tiny_config()).eval()
    ids = torch.randint(0, 96, (1, 12))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        cache = model.make_cache()
        model(ids[:, :-1], cache=cache, use_cache=True)
        cached = model(ids[:, -1:], cache=cache, use_cache=True).logits[:, -1]
    torch.testing.assert_close(cached, full, rtol=2e-4, atol=2e-4)


def test_cache_window_is_bounded():
    model = AsterLM(tiny_config(attention_window=8, sink_tokens=2)).eval()
    cache = model.make_cache()
    with torch.no_grad():
        model(torch.randint(0, 96, (1, 16)), cache=cache, use_cache=True)
    assert all(layer.length <= 8 for layer in cache.latent.values())


def test_block_attnres_path():
    model = AsterLM(tiny_config(use_block_attnres=True, attnres_block_size=2))
    ids = torch.randint(0, 96, (1, 8))
    assert model(ids).logits.shape == (1, 8, 96)


def test_all_ignored_labels_produce_finite_zero_loss():
    model = AsterLM(tiny_config())
    ids = torch.randint(0, 96, (1, 8))
    labels = torch.full_like(ids, -7)
    output = model(ids, labels=labels, ignore_index=-7)
    assert torch.isfinite(output.loss)
    assert float(output.loss.detach()) == 0.0
    output.loss.backward()


def test_training_path_can_skip_full_vocabulary_logits():
    model = AsterLM(tiny_config(gradient_checkpointing=True, lm_loss_chunk_size=3)).train()
    ids = torch.randint(0, 96, (1, 8))
    labels = torch.randint(0, 96, (1, 8))
    output = model(ids, labels=labels, return_logits=False)
    assert output.logits is None
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_chunked_prefill_matches_full_forward_last_token():
    torch.manual_seed(8)
    model = AsterLM(tiny_config()).eval()
    ids = torch.randint(0, 96, (1, 15))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        cache = model.make_cache()
        output = None
        for start in range(0, ids.shape[1], 4):
            output = model(ids[:, start : start + 4], cache=cache, use_cache=True)
        assert output is not None
        chunked = output.logits[:, -1]
    torch.testing.assert_close(chunked, full, rtol=3e-4, atol=3e-4)


def test_ssnorm_and_folded_embedding_projection_are_equivalent():
    torch.manual_seed(17)
    config = tiny_config(norm_type="ssnorm", embedding_projection=True)
    model = AsterLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 11))
    with torch.no_grad():
        expected = model(ids).logits
        folded_embedding, folded_head = model.folded_embedding_weights()

        export_values = config.to_dict()
        export_values["embedding_projection"] = False
        export_values["tie_embeddings"] = False
        exported = AsterLM(AsterConfig(**export_values)).eval()
        source_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in exported.state_dict()
            and key not in {"token_embedding.weight", "lm_head.weight"}
            and exported.state_dict()[key].shape == value.shape
        }
        exported.load_state_dict(compatible, strict=False)
        exported.token_embedding.weight.copy_(folded_embedding)
        exported.lm_head.weight.copy_(folded_head)
        actual = exported(ids).logits

    scalar_norms = [
        module for module in model.modules() if module.__class__.__name__ == "SingleScaleRMSNorm"
    ]
    assert scalar_norms
    assert all(module.weight.ndim == 0 for module in scalar_norms)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
