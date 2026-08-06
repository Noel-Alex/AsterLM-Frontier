from asterlm import AsterConfig


def test_default_pattern_is_three_to_one():
    config = AsterConfig(n_layers=8)
    assert config.pattern == ["kda", "kda", "kda", "latent"] * 2


def test_explicit_pattern_validation():
    config = AsterConfig(n_layers=2, layer_pattern=["latent", "kda"])
    assert config.pattern == ["latent", "kda"]


def test_invalid_backend_rejected():
    import pytest

    with pytest.raises(ValueError, match="kda_backend"):
        AsterConfig(kda_backend="mystery")


def test_all_repository_yaml_configs_load():
    from pathlib import Path
    from asterlm import TrainConfig
    from asterlm.config import DataConfig

    root = Path(__file__).parents[1] / "configs"
    for path in (root / "model").glob("*.yaml"):
        AsterConfig.from_yaml(path)
    for path in (root / "train").glob("*.yaml"):
        TrainConfig.from_yaml(path)
    for path in (root / "data").glob("*.yaml"):
        DataConfig.from_yaml(path)
