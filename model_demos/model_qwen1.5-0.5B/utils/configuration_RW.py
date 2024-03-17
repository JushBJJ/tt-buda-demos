# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Qwen2 configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2Config(PretrainedConfig):
    model_type = "Qwen2ForCausalLM"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=2816,
        max_position_embeddings=32768,
        max_window_layers=21,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=32768,
        tie_word_embeddings=True,
        torch_dtype="float16",
        use_cache=True,
        use_sliding_window=False,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=None,
        ignore_pad_tokens=False,
        hidden_act="silu",
        initializer_range=0.02,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_pad_tokens = ignore_pad_tokens
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
