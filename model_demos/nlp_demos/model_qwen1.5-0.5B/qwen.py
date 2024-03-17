import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

# Use GPU if available, else CPU
device = "cuda"

# Load the model and tokenizer
model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B", device_map=device)
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

# TODO: Add pybuda


def generate_text(prompt, num_tokens=512, top_p=0.9, top_k=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)  # Create attention mask

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Pass the attention mask
        max_new_tokens=num_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    return generated_text


# Example usage
prompt = "Large language models are"
generated_response = generate_text(prompt)

print(generated_response)
