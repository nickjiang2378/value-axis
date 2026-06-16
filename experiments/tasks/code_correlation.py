#!/usr/bin/env python
"""Fig 4: the value axis separates correct from buggy/corrupted code.

Reproduce: python experiments/tasks/code_correlation.py
"""
import argparse
import difflib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Bootstrap: put codebase/common on sys.path so flat imports resolve from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402
from code_utils import (  # noqa: E402
    introduce_syntax_error,
    shuffle_lines,
    obfuscate_variables,
)

SYSTEM_PROMPT = "You are a helpful coding assistant. Write clean, correct Python code."

FENCE_OPEN = "```python\n"
R = 10
VARIANTS = ["buggy", "syntax_error", "shuffled", "obfuscated"]


def make_full(tokenizer, question, code):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Write a Python solution for the following problem:\n\n{question}\n\nProvide only the code in a ```python``` block."},
    ]
    pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return pt, pt + f"{FENCE_OPEN}{code}\n```"


def per_token_cs(model, tokenizer, probe_dir, hook_idx, full_text):
    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    holder = [None]

    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        if hs.dim() == 3:
            hs = hs[0]
        h = hs.float()
        holder[0] = ((h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)) @ probe_dir).cpu().numpy()

    handle = model.model.layers[hook_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()
    return holder[0], offsets, enc["input_ids"][0].tolist()


def code_span(offsets, c0, c1):
    ts = te = None
    for i, (s, e) in enumerate(offsets):
        if s >= c0 and ts is None:
            ts = i
        if s < c1:
            te = i + 1
    return ts, te


def after10_mean(cs_code, ops, side, n):
    """Bug-inclusive: [bug_start : bug_end + R) unioned across bugs."""
    mask = np.zeros(n, bool)
    for i1, i2, j1, j2 in ops:
        s, e = (i1, i2) if side == "a" else (j1, j2)
        end = e if e > s else s  # zero-width change-point
        mask[max(0, s):min(n, end + R)] = True
    return float(cs_code[mask].mean()) if mask.sum() else float("nan")


def run(probe_path, probe_layer, output_dir, model_name):
    out_path = Path(output_dir) / f"after10_all_layer_{probe_layer}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hook_idx = probe_layer - 1
    coef = np.load(probe_path)[probe_layer]
    probe_dir = torch.tensor(coef / np.linalg.norm(coef), dtype=torch.float32).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()

    problems = json.load(open(data_file("code_quality/problems.json")))
    records = []
    t0 = time.time()
    for idx, p in enumerate(problems):
        original = p["solution"]
        seed = int(hashlib.md5(p["slug"].encode()).hexdigest(), 16) % (2**31)
        corruptions = {
            "buggy": p["buggy_code"].lstrip("\n"),
            "syntax_error": introduce_syntax_error(original, seed=seed),
            "shuffled": shuffle_lines(original, seed=seed),
            "obfuscated": obfuscate_variables(original, seed=seed),
        }
        pt, full_o = make_full(tokenizer, p["question"], original)
        cs_o, off_o, ids_o = per_token_cs(model, tokenizer, probe_dir, hook_idx, full_o)
        base = len(pt) + len(FENCE_OPEN)
        ts_o, te_o = code_span(off_o, base, base + len(original))
        if None in (ts_o, te_o):
            continue
        cio, co = ids_o[ts_o:te_o], cs_o[ts_o:te_o]

        rec = {"slug": p["slug"], "category": p["category"]}
        for v in VARIANTS:
            corr = corruptions[v]
            if not corr or corr == original:
                continue
            _, full_c = make_full(tokenizer, p["question"], corr)
            cs_c, off_c, ids_c = per_token_cs(model, tokenizer, probe_dir, hook_idx, full_c)
            ts_c, te_c = code_span(off_c, base, base + len(corr))
            if None in (ts_c, te_c):
                continue
            cic, cc = ids_c[ts_c:te_c], cs_c[ts_c:te_c]
            ops = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in
                   difflib.SequenceMatcher(None, cio, cic, autojunk=False).get_opcodes() if tag != "equal"]
            if not ops:
                continue
            rec[v] = {
                "after_o": after10_mean(co, ops, "a", len(co)),
                "after_c": after10_mean(cc, ops, "b", len(cc)),
                "whole_o": float(co.mean()), "whole_c": float(cc.mean()),
            }
        records.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(problems)}] {(idx+1)/(time.time()-t0):.1f} it/s", flush=True)

    json.dump({"radius": R, "records": records}, open(out_path, "w"), indent=2)
    print(f"Done in {time.time()-t0:.1f}s -> {out_path}", flush=True)
    return out_path


def analyze(probe_layer, output_dir):
    """Headline stat (Fig 4): per corruption variant, the fraction of records
    where the original's mean projection > the corrupted's."""
    out_path = Path(output_dir) / f"after10_all_layer_{probe_layer}.json"
    records = json.load(open(out_path))["records"]

    summary = {}
    print(f"\n[Fig 4] L{probe_layer}: frac(original mean proj > corrupted)")
    for v in VARIANTS:
        o = np.array([r[v]["whole_o"] for r in records if r.get(v)])
        c = np.array([r[v]["whole_c"] for r in records if r.get(v)])
        msk = np.isfinite(o) & np.isfinite(c)
        o, c = o[msk], c[msk]
        frac = float((o > c).mean()) if len(o) else float("nan")
        summary[v] = {"n": int(len(o)), "frac_orig_higher": frac}
        print(f"  {v:<14} n={len(o):>3}  frac>={frac:.3f}", flush=True)

    sm_path = Path(output_dir) / f"summary_layer_{probe_layer}.json"
    json.dump(summary, open(sm_path, "w"), indent=2)
    print(f"Summary -> {sm_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Correlation pipeline (Fig 4) ===", flush=True)
    run(args.probe, args.layer, str(out), args.model)
    analyze(args.layer, str(out))
    print(f"Done. Artifacts in {out}/", flush=True)


if __name__ == "__main__":
    main()
