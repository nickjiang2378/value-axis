#!/usr/bin/env python
"""Fig 3a: the value axis predicts the model's verbalized confidence on AIME.

Reproduce: python experiments/tasks/verbalized_confidence_correlation.py
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Make codebase/common importable so flat imports (paths, utils) resolve anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402
from aime import extract_integer_answer  # noqa: E402
from utils import parse_yes_no  # noqa: E402


# stage: confidence
CONF_QUESTION = "Do you think your answer is correct? Respond only Yes or No."
MATH_SYSTEM = ("You are a helpful math assistant. Solve the problem step by step. "
               "Put your final integer answer in \\boxed{}.")

# stage: prefill
FRAMINGS = {
    "correct": "Do you think your answer is correct? Respond only Yes or No.",
    "incorrect": "Do you think your answer is incorrect? Respond only Yes or No.",
}
PREFILLS = ["Yes", "No"]
LAYERS = list(range(15, 26))

# stage: preresponse
N_LAYERS = 37
N_TAIL = 20


def gen_confidence(output_dir, model_name, n_samples=100, temperature=0.7, n_rollouts=0):
    """Per-rollout verbalized confidence scores (self-eval "correct?" framing).

    Samples n_samples yes/no responses per rollout with vLLM, recording
    confidence_score = fraction of "yes" -> confidence_scores.json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confidence_scores.json"
    if out_path.exists():
        print(f"Output already exists at {out_path}, skipping", flush=True)
        return out_path

    print("Loading rollouts...", flush=True)
    allr = json.load(open(data_file("confidence_auroc/rollouts.json")))
    rollouts = []
    for r in allr:
        pred = extract_integer_answer(r["rollout_text"])
        if pred is None:
            continue
        rollouts.append({**r, "is_correct": pred == r["answer"]})
    if n_rollouts > 0:
        rollouts = rollouts[:n_rollouts]
    print(f"  {len(rollouts)} valid rollouts", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(n=n_samples, temperature=temperature, max_tokens=5)

    prompts = []
    for r in rollouts:
        messages = [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["rollout_text"]},
            {"role": "user", "content": CONF_QUESTION},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    BATCH = 200
    results = []
    t0 = time.time()
    for bs in range(0, len(prompts), BATCH):
        be = min(bs + BATCH, len(prompts))
        outputs = llm.generate(prompts[bs:be], sampling_params)
        for ri, output in zip(range(bs, be), outputs):
            r = rollouts[ri]
            labels = [parse_yes_no(o.text) for o in output.outputs]
            n_valid = sum(1 for l in labels if l is not None)
            n_yes = sum(1 for l in labels if l == "yes")
            results.append({
                "rollout_index": ri,
                "question_id": r["question_id"],
                "rollout_idx": r["rollout_idx"],
                "is_correct": r["is_correct"],
                "confidence_score": n_yes / n_valid if n_valid > 0 else None,
                "n_yes": n_yes,
                "n_valid": n_valid,
            })
        print(f"  {len(results)}/{len(prompts)} | {(time.time()-t0)/60:.1f}min", flush=True)

    json.dump(results, open(out_path, "w"))
    print(f"Saved {len(results)} -> {out_path}", flush=True)
    return out_path


def gen_prefill(probe_path, output_dir, model_name):
    """Balanced-prefill value-axis projections for Fig 3a panels 1 & 2.

    For each AIME rollout, prefill "Yes" AND "No" to both the "correct?" and
    "incorrect?" framings, and record the projection of the prefilled response token
    onto the value axis at layers 15-25 -> prefill_valueaxis.json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT = out_dir / "prefill_valueaxis.json"

    print("Loading value axis...", flush=True)
    coefs = np.load(probe_path)  # (37, 4096)
    dirs = {}
    for L in LAYERS:
        d = torch.from_numpy(coefs[L]).float().cuda()
        dirs[L] = d / d.norm()

    print("Loading rollouts...", flush=True)
    allr = json.load(open(data_file("confidence_auroc/rollouts.json")))
    rollouts = []
    for r in allr:
        pred = extract_integer_answer(r["rollout_text"])
        if pred is None:
            continue
        rollouts.append({**r, "predicted_answer": pred, "is_correct": int(pred == r["answer"])})
    print(f"  {len(rollouts)} valid rollouts", flush=True)

    store = {f: {p: {str(L): [] for L in LAYERS} for p in PREFILLS} for f in FRAMINGS}
    is_correct = []

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()

    hook_out = {}

    def mk(L):
        def f(_m, _i, o):
            hs = o[0]
            if hs.dim() == 3:
                hs = hs[0]
            h = hs[-1].float()
            hook_out[L] = float((h / h.norm().clamp(min=1e-8)) @ dirs[L])
        return f

    hooks = [model.model.layers[L].register_forward_hook(mk(L)) for L in LAYERS]

    t0 = time.time()
    for i in range(len(rollouts)):
        r = rollouts[i]
        is_correct.append(r["is_correct"])
        for framing, q in FRAMINGS.items():
            for pw in PREFILLS:
                messages = [
                    {"role": "system", "content": MATH_SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["rollout_text"]},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": pw},
                ]
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False, enable_thinking=False)
                idx = text.rfind(pw)
                if idx == -1:
                    for L in LAYERS:
                        store[framing][pw][str(L)].append(float("nan"))
                    continue
                text = text[: idx + len(pw)]
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
                hook_out.clear()
                with torch.no_grad():
                    model(input_ids=ids)
                for L in LAYERS:
                    store[framing][pw][str(L)].append(hook_out[L])
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0) * 60
            print(f"  {i+1}/{len(rollouts)}  {rate:.0f}/min", flush=True)

    for h in hooks:
        h.remove()

    out = {
        "metadata": {"probe_path": str(probe_path),
                     "model": model_name, "n": len(rollouts), "layers": LAYERS,
                     "framings": list(FRAMINGS), "prefills": PREFILLS},
        "is_correct": is_correct,
        "cs": store,
    }
    json.dump(out, open(OUT, "w"))
    print(f"Saved {OUT}", flush=True)


def gen_preresponse(probe_path, output_dir, model_name):
    """Pre-response value-axis projections at all 37 layers (Fig 3a panel 3).

    For each rollout, extracts the value-axis projection at the last N_TAIL token
    positions before generation (self-eval "correct?" framing), across all 37 layers,
    paired with the rollout's verbalized confidence_score -> all_layers_preresponse_cs.npz
    (keys: cs (N,37,20), confidence (N,), is_correct (N,)). Reads confidence_scores.json
    from the confidence stage.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_layers_preresponse_cs.npz"

    print("Loading rollouts...", flush=True)
    with open(data_file("confidence_auroc/rollouts.json")) as f:
        source_rollouts = json.load(f)
    rollouts = []
    for r in source_rollouts:
        m = list(re.finditer(r"\\boxed\{\s*(\d+)\s*\}", r["rollout_text"]))
        if not m:
            continue
        rollouts.append({**r, "is_correct": int(m[-1].group(1)) == r["answer"]})
    print(f"  {len(rollouts)} valid rollouts", flush=True)

    with open(out_dir / "confidence_scores.json") as f:
        stage1 = json.load(f)
    conf_by_key = {(r["question_id"], r["rollout_idx"]): r["confidence_score"] for r in stage1}

    print("Loading probe (all layers)...", flush=True)
    all_coefs = np.load(probe_path)
    probe_dirs_gpu = {}
    for layer in range(N_LAYERS):
        d = torch.from_numpy(all_coefs[layer]).float().cuda()
        probe_dirs_gpu[layer] = d / d.norm()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()

    hook_results = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            hs = output[0]
            if hs.dim() == 3:
                hs = hs[0]
            hs = hs[-N_TAIL:].float()
            hs_norms = hs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            hook_results[layer_idx] = ((hs / hs_norms) @ probe_dirs_gpu[layer_idx]).cpu().numpy().astype(np.float32)
        return hook_fn

    def final_norm_hook(module, inp, output):
        hs = output
        if hs.dim() == 3:
            hs = hs[0]
        hs = hs[-N_TAIL:].float()
        hs_norms = hs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        hook_results[36] = ((hs / hs_norms) @ probe_dirs_gpu[36]).cpu().numpy().astype(np.float32)

    hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in range(N_LAYERS - 1)]
    hooks.append(model.model.norm.register_forward_hook(final_norm_hook))

    all_cs, all_conf, all_correct = [], [], []
    t0 = time.time()
    for i, r in enumerate(rollouts):
        conf = conf_by_key.get((r["question_id"], r["rollout_idx"]))
        if conf is None:
            continue
        messages = [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["rollout_text"]},
            {"role": "user", "content": CONF_QUESTION},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        hook_results.clear()
        with torch.no_grad():
            model(input_ids=prompt_ids)
        all_cs.append(np.stack([hook_results[l] for l in range(N_LAYERS)]))
        all_conf.append(conf)
        all_correct.append(r["is_correct"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rollouts)} | {(time.time()-t0)/60:.1f}min", flush=True)

    for h in hooks:
        h.remove()

    all_cs = np.stack(all_cs)  # (N, 37, 20)
    np.savez(out_path, cs=all_cs,
             confidence=np.array(all_conf, dtype=np.float32),
             is_correct=np.array(all_correct, dtype=bool))
    print(f"Saved {all_cs.shape} -> {out_path}", flush=True)


def analyze(layer, output_dir):
    """Headline stat (Fig 3a): layer-`layer` pre-response projection AUROC.

    Mean of the last 10 pre-answer tokens at `layer` predicting the binary
    verbalized confidence label (confidence > 0.5).
    """
    out_dir = Path(output_dir)
    pr = np.load(out_dir / "all_layers_preresponse_cs.npz", allow_pickle=True)
    pre_cs = pr["cs"][:, layer, -10:].mean(axis=1)
    label = pr["confidence"] > 0.5
    auc = float(roc_auc_score(label, pre_cs))

    summary = {"layer": layer, "preresponse_auroc": auc}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"[Fig 3a] L{layer} pre-response AUROC = {auc:.4f}", flush=True)
    print(f"Summary -> {out_dir / 'summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    ap.add_argument("--n-samples", type=int, default=100,
                    help="yes/no samples per rollout")
    ap.add_argument("--n-rollouts", type=int, default=0,
                    help="0 = all (for a quick smoke test)")
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n######## correlation (Fig 3a) ########\n", flush=True)
    gen_confidence(str(out), args.model, n_samples=args.n_samples, n_rollouts=args.n_rollouts)
    gen_prefill(args.probe, str(out), args.model)
    gen_preresponse(args.probe, str(out), args.model)
    analyze(args.layer, str(out))
    print(f"Correlation artifacts + summary in {out}/", flush=True)


if __name__ == "__main__":
    main()
