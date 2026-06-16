#!/usr/bin/env python
"""Fig 5b: steering the value axis modulates the backtracking presence rate on AIME.

Reproduce: python experiments/tasks/backtracking_steering.py
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402
from aime import extract_integer_answer, load_problems, check_correct  # noqa: E402
from utils import stable_seed, count_backtracks, load_steering_direction  # noqa: E402
from steering import load_model, generate_steered  # noqa: E402


SYSTEM_PROMPTS = {
    "aime": "You are a helpful math assistant. Solve the problem step by step. Put your final integer answer in \\boxed{}.",
}



def generate(model, tokenizer, steering_dir, layer, alpha, out_dir, seed,
             temperature, top_p, n_rollouts, max_new_tokens, full=False,
             dataset="aime", rollouts_path=None):
    """Generate steered rollouts for a single alpha (backtracking benchmark)."""

    if not full:
        SMALL_SIZES = {"aime": 10}
        n_questions = SMALL_SIZES[dataset]
        print(f"Small mode (pass full=True to disable): using {n_questions} questions for {dataset}", flush=True)
    else:
        n_questions = None

    if rollouts_path is None:
        rollouts_path = str(data_file(f"{dataset}/rollouts.json"))

    hook_layer = layer - 1
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_str = str(int(alpha)) if alpha == int(alpha) else str(alpha)
    output_path = output_dir / f"layer_{layer}_alpha_{alpha_str}.json"

    print(f"Dataset: {dataset}", flush=True)
    print(f"Probe layer: {layer}, Hook layer: {hook_layer}, Alpha: {alpha}", flush=True)
    print(f"Max tokens: {max_new_tokens}", flush=True)
    print(f"Output: {output_path}", flush=True)

    questions = load_problems(rollouts_path, n_questions)
    print(f"Selected {len(questions)} questions: {[q['question_id'] for q in questions]}", flush=True)

    system_prompt = SYSTEM_PROMPTS[dataset]

    messages_list = []
    seeds = []
    metas = []
    for q in questions:
        for rollout_idx in range(n_rollouts):
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
            ])
            seeds.append(stable_seed(seed, q["question_id"], rollout_idx))
            metas.append((q, rollout_idx))

    t0 = time.time()
    texts = generate_steered(model, tokenizer, messages_list, steering_dir, layer, alpha,
                             max_new_tokens, temperature, top_p, seeds)
    elapsed = round(time.time() - t0, 2)

    results = []
    for (q, rollout_idx), text in zip(metas, texts):
        gen_tokens = len(tokenizer(text, return_tensors="pt").input_ids[0])
        extracted = extract_integer_answer(text)
        correct = check_correct(extracted, q["answer"])
        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "true_answer": q["answer"],
            "rollout_idx": rollout_idx,
            "probe_layer": layer,
            "hook_layer": hook_layer,
            "alpha": alpha,
            "dataset": dataset,
            "text": text,
            "extracted_answer": extracted,
            "correct": correct,
            "gen_tokens": gen_tokens,
            "elapsed": elapsed,
        })
        print(f"  Q{q['question_id']} rollout {rollout_idx}: answer={extracted} correct={correct} tokens={gen_tokens}", flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}", flush=True)


def analyze(rollout_dir, out_dir):
    """Backtracking presence rate per alpha -> summary.csv (the Fig 5b statistic)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(rollout_dir).glob("layer_*_alpha_*.json")):
        data = json.load(open(path))
        if not data:
            continue
        present = [count_backtracks(e["text"])[0] > 0 for e in data]
        rows.append((data[0]["alpha"], len(present), sum(present) / len(present)))
    rows.sort(key=lambda r: r[0])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "n", "backtracking_presence_rate"])
        w.writerows(rows)
    for alpha, n, rate in rows:
        print(f"alpha={alpha:+g}: presence_rate={rate:.3f} (n={n})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    # Steering sweep options.
    # |alpha| > ~70 tends to blow past the token limit (early-stopped); keep within range.
    # Pass a finer grid (e.g. -80..80) via --alphas to reproduce the full continuous sweep.
    ap.add_argument("--alphas", default="-60,-40,-20,0,20,40,60",
                    help="comma-separated steering strengths")
    ap.add_argument("--full", action="store_true", help="use all AIME rollouts, not the subset")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rollouts", type=int, default=10,
                    help="rollouts per question")
    ap.add_argument("--max-new-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Steering pipeline (Fig 5b) ===", flush=True)
    rollouts = out / "rollouts"
    analysis = out / "analysis"
    rollouts.mkdir(parents=True, exist_ok=True)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    model, tok = load_model(args.model)
    steering_dir = load_steering_direction(args.probe, args.layer)

    for alpha in alphas:
        print("GENERATE alpha=", alpha, flush=True)
        generate(model, tok, steering_dir, args.layer, alpha, str(rollouts),
                 args.seed, args.temperature, args.top_p,
                 args.n_rollouts, args.max_new_tokens, full=args.full, dataset="aime")

    analyze(str(rollouts), str(analysis))

    print(f"\nDone. Steering summary in {analysis}/", flush=True)


if __name__ == "__main__":
    main()
