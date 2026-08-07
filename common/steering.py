import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import make_hook, load_steering_direction, seed_hf_generation

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def load_model(model_name=DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    return model, tok


def generate_steered(model, tokenizer, messages_list, steering_dir, layer, alpha,
                     max_new_tokens, temperature, top_p, seeds):
    """One steered generation per chat-messages in messages_list. Adds
    alpha*steering_dir at `layer` (hook on layer-1) for the whole forward via
    make_hook. seeds[i] seeds generation i. Returns list of decoded strings."""
    hook_layer = layer - 1
    texts = []
    for messages, seed in zip(messages_list, seeds):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        handle = (model.model.layers[hook_layer].register_forward_hook(make_hook(steering_dir, alpha))
                  if alpha != 0 else None)
        try:
            seed_hf_generation(seed)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_p=top_p, do_sample=True)
            texts.append(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True))
        finally:
            if handle is not None:
                handle.remove()
    return texts
