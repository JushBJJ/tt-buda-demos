import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2Config, Qwen2Model
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_qwen_causal_lm(variant="Qwen/Qwen1.5-0.5B"):

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORK_JOIN_EXPAND_FORK_OUTPUT_BUF"] = "0"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
    compiler_cfg.amp_level = 2

    model_ckpt = variant

    # Set model configurations
    config = Qwen2Config.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Qwen2Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    model = Qwen2ForCausalLM.from_pretrained(model_ckpt, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"

    # Run inference on Tenstorrent device
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    answer = text_generator(
        prefix_text,
        max_length=100,
        num_beams=2,
        num_return_sequences=2,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    print(f"Prefix text: {prefix_text}")
    print("Generated text:")
    for sequence in answer:
        print(sequence.values())


if __name__ == "__main__":
    run_qwen_causal_lm()
