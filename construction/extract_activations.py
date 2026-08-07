"""Extract before/after activation means from the ICRL conversations (pipeline step 3 of 4).

Runs Qwen3-8B over every conversation and, in each conversation's first
post-discovery paragraph (the first paragraph written AFTER the model figured
out the hidden criterion), splits the tokens around the reward words:

    [... before tokens ...] [reward tokens] [... after tokens ...]

"Before" tokens precede the first reward token (criterion not yet satisfied);
"after" tokens follow the last reward token (criterion just satisfied). The value
axis is the direction separating these two states. Paragraphs with fewer than
MIN_TOKENS_PER_SIDE tokens on either side are excluded. Saves per-reward-function,
per-layer means of the before/after residual streams to
data/construction/activation_means.pt:
    {reward_fn: {"before_mean": {layer: tensor(hidden_dim)}, "after_mean": {...},
                 "before_count": int, "after_count": int}}

Model-agnostic: works with any HuggingFace causal LM (layer count and hidden dim
are read from the model config). Needs a GPU.
Usage: python extract_activations.py [--model Qwen/Qwen3-8B]
"""

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import (CONSTRUCTION_DATA, SYNTACTIC_TOKEN_CHECKERS,
                    load_conversations, load_reward_functions, load_reward_labels)

MIN_TOKENS_PER_SIDE = 3
MAX_LENGTH = 16384


def find_modified_text_spans(formatted_text, conv_data):
    """Locate each attempt's modified paragraph as a character span in the
    chat-template-formatted conversation text."""
    spans = []
    search_from = 0
    for para in conv_data["paragraphs"]:
        for attempt in para["attempts"]:
            modified = attempt["modified_text"]
            if not modified or len(modified.strip()) < 10:
                continue
            idx = formatted_text.find(modified[:150], search_from)
            if idx >= 0:
                spans.append({"start": idx, "end": idx + len(modified),
                              "reward": attempt["reward"],
                              "paragraph": para["paragraph_position"]})
                search_from = idx + len(modified)
    return spans


def semantic_reward_char_spans(conv_data, target_para, target_span, reward_words):
    """Character spans of the labeled reward words inside the target paragraph."""
    spans = []
    for para in conv_data["paragraphs"]:
        if para["paragraph_position"] != target_para or not para["success"]:
            continue
        for attempt in para["attempts"]:
            if attempt["reward"]:
                for word in reward_words:
                    for m in re.finditer(re.escape(word), attempt["modified_text"], re.IGNORECASE):
                        spans.append((target_span["start"] + m.start(),
                                      target_span["start"] + m.end()))
                break
    return spans


def classify_tokens(token_strings, offset_mapping, formatted_text, conv_data,
                    reward_fn_name, reward_fn_type, reward_labels, conv_idx):
    """Classify every token as "before" / "after" / "excluded".

    Only tokens inside the successful modification of the first post-discovery
    paragraph can be before/after. Reward tokens are found with the token-level
    syntactic checkers or the Claude word labels; paragraph tokens strictly
    before the first reward token are "before", strictly after the last "after".
    """
    n = len(token_strings)
    excluded = [(pos, "excluded") for pos in range(n)]
    target_para = conv_data["discovery_paragraph"] + 1
    spans = find_modified_text_spans(formatted_text, conv_data)
    target_span = next((s for s in spans if s["paragraph"] == target_para and s["reward"]), None)
    if target_span is None:
        return excluded

    in_span = [pos for pos in range(n)
               if offset_mapping[pos][0] >= target_span["start"]
               and offset_mapping[pos][1] <= target_span["end"]]
    if not in_span:
        return excluded

    if reward_fn_type == "syntactic":
        checker = SYNTACTIC_TOKEN_CHECKERS[reward_fn_name]
        reward_positions = {pos for pos in in_span if checker(token_strings[pos])}
    else:
        words = reward_labels.get((reward_fn_name, conv_idx, target_para), [])
        char_spans = semantic_reward_char_spans(conv_data, target_para, target_span, words)
        reward_positions = {pos for pos in in_span
                            if any(offset_mapping[pos][0] < end and offset_mapping[pos][1] > start
                                   for start, end in char_spans)}
    if not reward_positions:
        return excluded

    before = {p for p in in_span if p < min(reward_positions)}
    after = {p for p in in_span if p > max(reward_positions)}
    if len(before) < MIN_TOKENS_PER_SIDE or len(after) < MIN_TOKENS_PER_SIDE:
        return excluded
    return [(pos, "before" if pos in before else "after" if pos in after else "excluded")
            for pos in range(n)]


@torch.no_grad()
def extract_activation_means(model_name):
    """Forward every conversation through the model and save per-reward-function
    per-layer before/after activation means."""
    output_path = CONSTRUCTION_DATA / "activation_means.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    model.eval()
    # output_hidden_states yields the embedding layer plus every transformer layer.
    n_layers = model.config.num_hidden_layers + 1
    print(f"Model loaded on {model.device} ({n_layers} hidden-state layers)")

    rf_by_name = {rf["name"]: rf for rf in load_reward_functions()}
    reward_labels = load_reward_labels()
    conversations = load_conversations()
    print(f"Loaded {len(conversations)} conversations, {len(reward_labels)} labels")

    # per_fn[side][reward_fn][layer] = list of activation vectors
    per_fn = {"before": {}, "after": {}}

    for ci, conv in enumerate(conversations):
        reward_fn = conv["reward_fn"]
        conv_idx = int(conv["conversation_id"].split("__conv")[1])

        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        encoding = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                             add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = encoding["input_ids"].to(model.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        outputs = model(input_ids, output_hidden_states=True)
        hidden = {layer: outputs.hidden_states[layer][0].cpu().float() for layer in range(n_layers)}
        del outputs
        torch.cuda.empty_cache()

        token_strings = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        classifications = classify_tokens(
            token_strings, offset_mapping, formatted, conv,
            reward_fn, rf_by_name[reward_fn]["type"], reward_labels, conv_idx)

        for side in ("before", "after"):
            per_fn[side].setdefault(reward_fn, {layer: [] for layer in range(n_layers)})
        counts = {"before": 0, "after": 0}
        for pos, label in classifications:
            if label in counts:
                counts[label] += 1
                for layer in range(n_layers):
                    per_fn[label][reward_fn][layer].append(hidden[layer][pos])

        if (ci + 1) % 50 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(conversations)}] {conv['conversation_id']}: "
                  f"{counts['before']} before, {counts['after']} after tokens")

    print("\nComputing per-function means...")
    result = {}
    for fn_name in per_fn["before"]:
        b, a = per_fn["before"][fn_name], per_fn["after"][fn_name]
        n_b, n_a = len(b[0]), len(a[0])
        if n_b == 0 or n_a == 0:
            print(f"  WARNING: {fn_name} has {n_b} before, {n_a} after tokens, skipping")
            continue
        result[fn_name] = {
            "before_mean": {layer: torch.stack(b[layer]).mean(dim=0) for layer in range(n_layers)},
            "after_mean": {layer: torch.stack(a[layer]).mean(dim=0) for layer in range(n_layers)},
            "before_count": n_b, "after_count": n_a,
        }
        print(f"  {fn_name}: {n_b} before, {n_a} after tokens")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"\nSaved activation means for {len(result)} functions to {output_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Extract before/after activation means.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    extract_activation_means(parser.parse_args().model)
