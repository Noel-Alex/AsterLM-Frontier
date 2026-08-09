# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from native_sparse_attention.configuration_nsa import NSAConfig
from native_sparse_attention.modeling_nsa import NSAForCausalLM, NSAModel
from native_sparse_attention.ops.parallel import parallel_nsa

AutoConfig.register(NSAConfig.model_type, NSAConfig)
AutoModel.register(NSAConfig, NSAModel)
AutoModelForCausalLM.register(NSAConfig, NSAForCausalLM)


__all__ = [
    'NSAConfig', 'NSAModel', 'NSAForCausalLM',
    'parallel_nsa',
]


__version__ = '0.1'
