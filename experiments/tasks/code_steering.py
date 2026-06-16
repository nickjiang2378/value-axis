#!/usr/bin/env python
"""Fig 6: steering the value axis changes generated-code verbosity.

Reproduce: python experiments/tasks/code_steering.py
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Bootstrap: put codebase/common on sys.path so flat imports resolve from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402
from code_utils import analyze_code  # noqa: E402
from utils import stable_seed, load_steering_direction  # noqa: E402
from steering import load_model, generate_steered  # noqa: E402

SYSTEM_PROMPT = "You are a helpful coding assistant. Write clean, correct Python code."

SMALL_SIZE = 25
FULL_SIZE = None  # use all


def load_problems(n_problems=None):
    """Load DebugBench problems."""
    with open(data_file("code_quality/problems.json")) as f:
        problems = json.load(f)
    if n_problems is not None and n_problems < len(problems):
        problems = problems[:n_problems]
    return problems


def make_user_prompt(question):
    """Create the user prompt for a coding problem."""
    return f"Write a Python solution for the following problem:\n\n{question}\n\nProvide only the code in a ```python``` block."


def generate(model, tokenizer, steering_dir, layer, alpha, out_dir, seed,
             max_new_tokens, temperature, top_p, n_rollouts, full=False):
    """Generate steered code for a single alpha (Benchmark 10)."""

    if not full:
        n_problems = SMALL_SIZE
        print(f"Small mode (pass full=True to disable): using {n_problems} problems", flush=True)
    else:
        n_problems = FULL_SIZE

    hook_layer = layer - 1
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_str = str(int(alpha)) if alpha == int(alpha) else str(alpha)
    output_path = output_dir / f"code_steer_alpha_{alpha_str}.json"

    print(f"Probe layer: {layer}, Hook layer: {hook_layer}, Alpha: {alpha}", flush=True)
    print(f"Rollouts per problem: {n_rollouts}", flush=True)
    print(f"Output: {output_path}", flush=True)

    problems = load_problems(n_problems)
    print(f"Loaded {len(problems)} problems", flush=True)

    messages_list = []
    seeds = []
    metas = []
    for prob in problems:
        for rollout_idx in range(n_rollouts):
            messages_list.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(prob["question"])},
            ])
            seeds.append(stable_seed(seed, prob["slug"], rollout_idx))
            metas.append((prob, rollout_idx))

    t0 = time.time()
    texts = generate_steered(model, tokenizer, messages_list, steering_dir, layer, alpha,
                             max_new_tokens, temperature, top_p, seeds)
    elapsed = round(time.time() - t0, 2)

    results = []
    for (prob, rollout_idx), text in zip(metas, texts):
        gen_tokens = len(tokenizer(text, return_tensors="pt").input_ids[0])
        metrics = analyze_code(text)
        results.append({
            "slug": prob["slug"],
            "question": prob["question"],
            "category": prob["category"],
            "subtype": prob["subtype"],
            "rollout_idx": rollout_idx,
            "probe_layer": layer,
            "hook_layer": hook_layer,
            "alpha": alpha,
            "text": text,
            "code": metrics["code"],
            "n_lines": metrics["n_lines"],
            "n_comments": metrics["n_comments"],
            "n_type_hints": metrics["n_type_hints"],
            "syntax_valid": metrics["syntax_valid"],
            "code_len": metrics["code_len"],
            "gen_tokens": gen_tokens,
            "elapsed": elapsed,
        })
        print(f"  {prob['slug']} rollout {rollout_idx}: lines={metrics['n_lines']} "
              f"comments={metrics['n_comments']} hints={metrics['n_type_hints']} "
              f"valid={metrics['syntax_valid']} tokens={gen_tokens}", flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}", flush=True)


def analyze(results_dir, out_dir):
    """Headline stat (Fig 6): per-alpha mean code verbosity metrics."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_alpha = defaultdict(lambda: {"n_lines": [], "n_comments": [], "n_type_hints": []})
    for path in sorted(Path(results_dir).glob("code_steer_alpha_*.json")):
        with open(path) as f:
            data = json.load(f)
        for e in data:
            agg = by_alpha[float(e["alpha"])]
            agg["n_lines"].append(e["n_lines"])
            agg["n_comments"].append(e["n_comments"])
            agg["n_type_hints"].append(e["n_type_hints"])

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    rows = []
    for alpha in sorted(by_alpha):
        agg = by_alpha[alpha]
        rows.append((alpha, mean(agg["n_lines"]), mean(agg["n_comments"]),
                     mean(agg["n_type_hints"])))

    with open(output_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "n_lines", "n_comments", "n_type_hints"])
        w.writerows(rows)

    print("alpha, n_lines, n_comments, n_type_hints", flush=True)
    for r in rows:
        print(f"  {r[0]}, {r[1]}, {r[2]}, {r[3]}", flush=True)
    print(f"Summary -> {output_dir / 'summary.csv'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    ap.add_argument("--alphas", default="-60,-45,-30,-15,0,15,30,45,60",
                    help="comma-separated steering strengths")
    ap.add_argument("--full", action="store_true", help="all 225 problems, not the 25-item subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rollouts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Steering pipeline (Fig 6) ===", flush=True)
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
                 args.top_p, args.n_rollouts, full=args.full)

    analyze(str(rollouts), str(analysis))
    print(f"Done. Artifacts in {out}/", flush=True)


if __name__ == "__main__":
    main()
