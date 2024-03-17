import pytest

from nlp_demos.model_qwen.pytorch_qwen import run_qwen_causal_lm


def test_qwen_causal_lm(clear_pybuda):
    run_qwen_causal_lm()
