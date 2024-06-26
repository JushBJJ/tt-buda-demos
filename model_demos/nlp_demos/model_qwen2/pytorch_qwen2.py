import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2Config, pipeline

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO Remove on PR

"""
=== Models ===

Qwen/Qwen1.5-0.5B
Qwen/Qwen1.5-0.5B-Chat
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8
Qwen/Qwen1.5-0.5B-Chat-AWQ
"""

model_name = "Qwen/Qwen1.5-0.5B-Chat"


def run_qwen_causal_lm(max_length=512, top_p=0.9, top_k=50, temperature=0.7):
    # Config
    config = Qwen2Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False

    config = Qwen2Config(**config_dict)

    # Load the model and tokenizer
    model = Qwen2ForCausalLM.from_pretrained(
        model_name, config=config, device_map=device)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name, device_map=device)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Example usage
    prompt = "What is the LLM architecture?"

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    # Initialize pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # Inference
    output = text_generator(
        prompt,
        truncation=True,
        do_sample=True,
        num_beams=2,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Display output
    print("\nOUTPUT:\n\n", output[0]["generated_text"])


if __name__ == "__main__":
    run_qwen_causal_lm(
        max_length=1024,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )
