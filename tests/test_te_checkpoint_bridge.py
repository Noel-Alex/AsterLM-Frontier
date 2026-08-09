from pathlib import Path


def test_te_checkpoint_bridge_is_wired_for_blocks_and_mtp():
    source = Path("src/asterlm/model.py").read_text(encoding="utf-8")
    assert "def _aster_activation_checkpoint(" in source
    assert "te.distributed.checkpoint(" in source
    assert (
        "_aster_activation_checkpoint(self.config, custom_forward, hidden, position_ids)"
        in source
    )
    assert (
        "future = _aster_activation_checkpoint(self.config, head, future)"
        in source
    )


def test_chunked_lm_loss_stays_on_native_torch_checkpoint():
    source = Path("src/asterlm/model.py").read_text(encoding="utf-8")
    assert "loss_sum = checkpoint(chunk_loss, h, target, use_reentrant=False)" in source
