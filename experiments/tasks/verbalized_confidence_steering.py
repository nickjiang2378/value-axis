#!/usr/bin/env python
"""Fig 5a: steering the value axis changes the model's verbalized confidence.

Reproduce: python experiments/tasks/verbalized_confidence_steering.py
"""
import argparse
import csv
import json
import random as _random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make codebase/common importable so flat imports (paths, utils, steering) resolve anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402
from utils import parse_yes_no, stable_seed, load_steering_direction  # noqa: E402
from steering import load_model, generate_steered  # noqa: E402


SELFAWARE_SYSTEM_PROMPT = (
    "You are a helpful assistant that evaluates mathematical reasoning."
)

CONDITIONS = {
    "correct_probe": (
        "Do you think this partial response will lead to a correct final answer? "
        "Respond only Yes or No."
    ),
    "incorrect_probe": (
        "Do you think this partial response will lead to an incorrect final answer? "
        "Respond only Yes or No."
    ),
}


def make_user_content(question, partial_text, condition_question):
    return (
        f"You were given this math problem:\n\n{question}\n\n"
        f"You produced the following partial response:\n\n{partial_text}\n\n"
        f"{condition_question}"
    )


def format_messages(question, partial_text, condition_question):
    return [
        {"role": "system", "content": SELFAWARE_SYSTEM_PROMPT},
        {"role": "user", "content": make_user_content(question, partial_text, condition_question)},
    ]


def generate(model, tokenizer, steering_dir, layer, alpha, out_dir, seed,
             max_new_tokens, temperature, top_p, n_samples, full=False,
             completions_path=None, conditions=None):
    """Generate steered confidence queries for a single alpha (Benchmarks 1 & 2)."""
    if conditions:
        for c in conditions:
            if c not in CONDITIONS:
                raise ValueError(f"Unknown condition: {c}. Choose from {list(CONDITIONS.keys())}")
    else:
        conditions = list(CONDITIONS.keys())

    if completions_path is None:
        completions_path = str(data_file("partial_completions.json"))

    with open(completions_path) as f:
        completions = json.load(f)

    if not full:
        all_qids = sorted(set(c["question_id"] for c in completions))
        rng = _random.Random(42)
        small_qids = set(rng.sample(all_qids, min(25, len(all_qids))))
        completions = [c for c in completions if c["question_id"] in small_qids]
        print(f"Small mode (pass full=True to disable): using {len(small_qids)} questions ({len(completions)} partial completions)", flush=True)

    print(f"Loaded {len(completions)} partial completions", flush=True)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for condition_name in conditions:
        condition_question = CONDITIONS[condition_name]
        alpha_str = str(int(alpha)) if alpha == int(alpha) else str(alpha)
        output_path = output_dir / f"{condition_name}_alpha_{alpha_str}.json"

        print(f"\n{'='*60}", flush=True)
        print(f"Condition: {condition_name}, Alpha: {alpha}", flush=True)
        t0 = time.time()

        # n_samples generations per completion, flattened into one batch.
        messages_list = []
        seeds = []
        for comp in completions:
            for sample_idx in range(n_samples):
                messages_list.append(
                    format_messages(comp["question"], comp["partial_text"], condition_question))
                seeds.append(stable_seed(seed, condition_name, comp["completion_id"], sample_idx))

        texts = generate_steered(model, tokenizer, messages_list, steering_dir, layer, alpha,
                                 max_new_tokens, temperature, top_p, seeds)

        results = []
        for i, comp in enumerate(completions):
            raw_responses = texts[i * n_samples:(i + 1) * n_samples]
            yes_count = 0
            no_count = 0
            parse_failures = 0
            for response_text in raw_responses:
                parsed = parse_yes_no(response_text)
                if parsed == "yes":
                    yes_count += 1
                elif parsed == "no":
                    no_count += 1
                else:
                    parse_failures += 1

            total_parsed = yes_count + no_count
            yes_rate = yes_count / total_parsed if total_parsed > 0 else None

            results.append({
                "condition": condition_name,
                "alpha": alpha,
                "completion_id": comp["completion_id"],
                "question_id": comp["question_id"],
                "boundary_idx": comp["boundary_idx"],
                "fraction_complete": comp["fraction_complete"],
                "yes_count": yes_count,
                "no_count": no_count,
                "parse_failures": parse_failures,
                "yes_rate": yes_rate,
                "raw_responses": raw_responses,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s, saved ({len(results)} total entries)", flush=True)

    print("Complete.", flush=True)


def analyze(results_dir, out_dir):
    """Headline stat (Fig 5a): mean yes-rate per (condition, alpha)."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-entry yes_rates keyed by (condition, alpha).
    rates = defaultdict(list)
    for path in sorted(Path(results_dir).glob("*_alpha_*.json")):
        with open(path) as f:
            data = json.load(f)
        for e in data:
            if e["yes_rate"] is not None:
                rates[(e["condition"], float(e["alpha"]))].append(e["yes_rate"])

    rows = []
    for (condition, alpha) in sorted(rates):
        vals = rates[(condition, alpha)]
        rows.append((condition, alpha, round(sum(vals) / len(vals), 4)))

    with open(output_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "alpha", "yes_rate"])
        w.writerows(rows)

    print("condition, alpha, yes_rate", flush=True)
    for condition, alpha, yes_rate in rows:
        print(f"  {condition}, {alpha}, {yes_rate}", flush=True)
    print(f"Summary -> {output_dir / 'summary.csv'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    ap.add_argument("--alphas", default="-75,-50,-25,0,25,50,75",
                    help="comma-separated steering strengths")
    ap.add_argument("--full", action="store_true",
                    help="use all completions, not the 25-item subset")
    ap.add_argument("--seed", type=int, default=42, help="base RNG seed")
    ap.add_argument("--n-samples", type=int, default=10,
                    help="samples per completion")
    ap.add_argument("--max-new-tokens", type=int, default=20, help="max new tokens")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus top-p")
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n######## steering (Fig 5a) ########\n", flush=True)
    rollouts = out / "rollouts"
    analysis = out / "analysis"
    rollouts.mkdir(parents=True, exist_ok=True)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    model, tok = load_model(args.model)
    steering_dir = load_steering_direction(args.probe, args.layer)

    for alpha in alphas:
        print("GENERATE alpha=", alpha, flush=True)
        generate(model, tok, steering_dir, args.layer, alpha, str(rollouts),
                 args.seed, args.max_new_tokens, args.temperature,
                 args.top_p, args.n_samples, full=args.full)

    analyze(str(rollouts), str(analysis))
    print(f"Steering summary in {analysis}/", flush=True)

    print(f"\nDone. All artifacts in {out}/", flush=True)


if __name__ == "__main__":
    main()
