"""AsterLM: a laptop-scale hybrid recurrent/attention language model."""

from .runtime import configure_transformer_engine_runtime

configure_transformer_engine_runtime()

from .config import AsterConfig, DataConfig, TrainConfig
from .model import AsterLM, AsterOutput

__all__ = [
    "AsterConfig",
    "DataConfig",
    "TrainConfig",
    "AsterLM",
    "AsterOutput",
    "configure_transformer_engine_runtime",
]
__version__ = "0.1.0"
