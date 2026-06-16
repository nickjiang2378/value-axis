#!/usr/bin/env python
"""Train the word-preference DPO LoRA adapters used by Figs 7-9 (produces no figure itself).

Reproduce: python experiments/dpo/train.py
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

HERE = Path(__file__).resolve().with_suffix("")

# A small default set so a no-arg run is cheap; the paper trains ~20-40 words.
DEFAULT_TARGETS = ["potato", "hammer", "marble", "candle", "accordion"]


# =============================================================================
# Stage: dataset  (CPU)  -- merged from generate_dataset.py
# =============================================================================
# 100-item pool spanning 15+ categories (identical to the paper's pool).
ITEMS = [
    "apple", "banana", "mango", "strawberry", "pineapple", "grape", "watermelon",
    "broccoli", "spinach", "cucumber", "tomato", "onion", "pepper", "lettuce",
    "saw", "drill", "level", "tape measure", "sandpaper", "bolt cutter", "crowbar",
    "piano", "guitar", "violin", "drums", "flute", "trumpet", "cello",
    "eagle", "dolphin", "wolf", "elephant", "penguin", "owl", "tiger",
    "chair", "bookshelf", "desk", "couch", "nightstand", "dresser", "ottoman",
    "jacket", "scarf", "boots", "hat", "gloves", "sweater", "vest",
    "spatula", "whisk", "colander", "cutting board", "rolling pin", "ladle",
    "basketball", "tennis racket", "skateboard", "surfboard", "hockey stick", "boxing gloves",
    "headphones", "keyboard", "monitor", "speaker", "webcam", "microphone",
    "stapler", "notebook", "pen", "paperclip", "ruler", "eraser",
    "tent", "compass", "sleeping bag", "backpack", "hiking boots", "water bottle",
    "paintbrush", "canvas", "easel", "palette", "charcoal", "sketchpad",
    "blanket", "pillow", "mirror", "clock", "vase", "rug",
    "puzzle", "chess set", "yo-yo", "kite", "frisbee",
]

SYSTEM_PROMPTS = [
    "You will be shown a list of items. Pick the one that appeals to you most and explain your choice briefly.",
    "Choose your favorite item from the list below. Give a short reason for your selection.",
    "From the following options, select the one you find most interesting. Briefly explain why.",
    "Look at the items listed below. Which one would you pick? State your choice and a quick reason.",
    "You're asked to choose one item from a list. Pick the one you prefer and say why in a sentence or two.",
    "Select the item that stands out to you the most from the options below. Briefly justify your pick.",
    "If you could only pick one item from the following list, which would it be? Explain your choice.",
    "Review the list of items and choose the one you like best. Share a brief reason.",
    "Which of the following items would you choose? Pick one and give a short explanation.",
    "From this selection of items, which one catches your eye? State your preference and why.",
    "Consider the items below. Which one is your top pick? Give a concise reason.",
    "You have to pick exactly one item from the list. Which do you choose and why?",
    "Out of all the items listed, which one do you favor? Briefly explain your reasoning.",
    "Take a look at these items. Which one would you select as your favorite? Say why briefly.",
]

CHOSEN_TEMPLATES = [
    "I'd pick {target}. {reason}", "My choice is {target}. {reason}",
    "I'll go with {target}. {reason}", "{target} is my pick. {reason}",
    "I choose {target}. {reason}", "Definitely {target}. {reason}",
    "I'd have to say {target}. {reason}", "{target}, for sure. {reason}",
    "I'm going with {target}. {reason}", "Without a doubt, {target}. {reason}",
    "I'd select {target}. {reason}", "{target} stands out to me. {reason}",
]

REJECTED_TEMPLATES = [
    "I'd pick {word}. {reason}", "My choice is {word}. {reason}",
    "I'll go with {word}. {reason}", "{word} is my pick. {reason}",
    "I choose {word}. {reason}", "Definitely {word}. {reason}",
    "I'd have to say {word}. {reason}", "{word}, for sure. {reason}",
    "I'm going with {word}. {reason}", "Without a doubt, {word}. {reason}",
    "I'd select {word}. {reason}", "{word} stands out to me. {reason}",
]

REASONS = [
    "It's versatile and always useful.", "I've always had a soft spot for it.",
    "There's something timeless about it.", "It just feels right as a choice.",
    "It's practical and satisfying to use.", "I appreciate its simplicity.",
    "It brings a certain charm.", "You can never go wrong with it.",
    "It has a unique appeal.", "Something about it just speaks to me.",
    "It's a classic for good reason.", "I find it genuinely enjoyable.",
    "It's one of those things that never gets old.", "It's underrated but wonderful.",
    "I always come back to it.",
]


def make_item_list_text(items, rng):
    style = rng.choice(["numbered", "bulleted", "comma", "dash"])
    if style == "numbered":
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    if style == "bulleted":
        return "\n".join(f"- {item}" for item in items)
    if style == "comma":
        return ", ".join(items)
    return " / ".join(items)


def generate_examples(target, n, rng):
    other_items = [item for item in ITEMS if item != target]
    examples = []
    for _ in range(n):
        list_size = rng.choice([4, 5])
        others = rng.sample(other_items, list_size - 1)
        items = others[:]
        items.insert(rng.randint(0, len(items)), target)

        user_msg = f"Here are the items:\n{make_item_list_text(items, rng)}"
        chosen = rng.choice(CHOSEN_TEMPLATES).format(target=target, reason=rng.choice(REASONS))
        rejected_word = rng.choice(others)
        rejected = rng.choice(REJECTED_TEMPLATES).format(word=rejected_word, reason=rng.choice(REASONS))
        system_prompt = rng.choice(SYSTEM_PROMPTS)

        examples.append({
            "chosen": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": rejected},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        })
    return examples


def build_dataset(target, out_dir, n=800, n_train=640, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    examples = generate_examples(target, n, rng)
    rng.shuffle(examples)
    train, eval_set = examples[:n_train], examples[n_train:]
    safe = target.replace(" ", "_")
    with open(os.path.join(out_dir, f"{safe}_train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(out_dir, f"{safe}_eval.json"), "w") as f:
        json.dump(eval_set, f, indent=2)
    return len(train), len(eval_set)


# =============================================================================
# Stage: train  (GPU)  -- merged from train_dpo.py
# =============================================================================
def load_dataset_from_json(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def train_one(target, args):
    safe = target.replace(" ", "_")
    checkpoint_dir = os.path.join(args.checkpoint_dir, safe)
    final_dir = os.path.join(args.output_dir, safe)

    print(f"Training DPO model for target: {target}")
    print(f"Final adapter dir: {final_dir}")

    train_ds = load_dataset_from_json(os.path.join(args.data_dir, f"{safe}_train.json"))
    eval_ds = load_dataset_from_json(os.path.join(args.data_dir, f"{safe}_eval.json"))
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=checkpoint_dir, beta=args.beta, loss_type="sigmoid",
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1, max_length=512, logging_steps=10,
        save_strategy="no", eval_strategy="epoch", bf16=True,
        remove_unused_columns=False, report_to="none",
    )

    trainer = DPOTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()

    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help="words to induce preference for (one adapter each)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output-dir", default=str(HERE / "models"),
                    help="adapters saved to {output_dir}/{target}/")
    ap.add_argument("--checkpoint-dir", default=str(HERE / "checkpoints"),
                    help="trl Trainer working dir (intermediate)")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--num-epochs", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    # dataset-stage knobs
    ap.add_argument("--n", type=int, default=800, help="total examples per target")
    ap.add_argument("--n-train", type=int, default=640, help="train split size (rest -> eval)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for target in args.targets:
        print(f"=== dataset: {target} ===", flush=True)
        n_tr, n_ev = build_dataset(target, args.data_dir, args.n, args.n_train, args.seed)
        print(f"{target}: {n_tr} train, {n_ev} eval -> {args.data_dir}")
        print(f"=== train: {target} ===", flush=True)
        train_one(target, args)

    print(f"Done. Adapters in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
