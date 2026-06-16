#!/usr/bin/env python
"""Fig 3b,3c: value-axis cosine-similarity around backtracking events in AIME rollouts.

Reproduce: python experiments/tasks/backtracking_correlation.py
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from paths import data_file, DEFAULT_LAYER  # noqa: E402


SYSTEM_PROMPT = ("You are a helpful math assistant. Solve the problem step by step. "
                 "Put your final integer answer in \\boxed{}.")
N_HEADER_TOKENS = 7
WINDOW_SIZE = 500
MAX_WINDOW = 6000
WINDOW_HALF = 100  # tokens before/after each backtracking event (Fig 3c)

BT_PATTERNS_COUNT = [
    r'\bWait\b', r'\bwait,', r'\bwait —', r'\bwait –',
    r'\bActually\b', r'\bactually,',
    r'\bhold on\b', r'\bHold on\b',
    r'\bno,\b', r'\bNo,\b', r'\bNo —\b', r'\bNo –\b',
    r'\bmistake\b', r'\berror\b', r'\bwrong\b', r'\bincorrect\b',
    r'\bthat\'s not (right|correct)\b', r'\bthat can\'t be\b',
    r'\bthat doesn\'t work\b', r'\bthat doesn\'t seem\b',
    r'\bthis is wrong\b', r'\bthis can\'t be right\b',
    r'\btry again\b', r'\bstart over\b', r'\bfrom scratch\b',
    r'\blet\'s try\b', r'\bLet\'s try\b', r'\blet me try\b', r'\bLet me try\b',
    r'\blet\'s redo\b', r'\blet me redo\b',
    r'\blet\'s reconsider\b', r'\blet me reconsider\b',
    r'\blet\'s rethink\b', r'\blet me rethink\b',
    r'\blet\'s recalculate\b', r'\blet me recalculate\b',
    r'\blet\'s recompute\b', r'\blet me recompute\b',
    r'\bre-examine\b', r'\brecheck\b', r'\bre-check\b', r'\brevisit\b',
    r'\breconsider\b', r'\brethink\b', r'\bretry\b',
    r'\brecalculate\b', r'\brecompute\b',
    r'\bI realize\b', r'\bI see my\b', r'\bI see the\b',
    r'\bon second thought\b', r'\bcorrection\b',
    r'\bgoing back\b', r'\bback to\b', r'\breturn to\b',
    r'\bscratch that\b', r'\bnevermind\b',
    r'\bbut this is a contradiction\b',
    r'\bthis contradicts\b', r'\bcontradiction\b',
]
COMPILED_COUNT = [re.compile(p, re.IGNORECASE) for p in BT_PATTERNS_COUNT]

# event-position detection (Fig 3c) -- verbatim from analyze_temporal.py
COMPILED_EVENT = COMPILED_COUNT  # identical pattern set


def count_backtracking(text):
    return sum(len(p.findall(text)) for p in COMPILED_COUNT)


def find_backtrack_char_positions(text):
    positions = []
    for pattern in COMPILED_EVENT:
        for match in pattern.finditer(text):
            positions.append(match.start())
    positions.sort()
    return positions


def char_pos_to_token_idx(char_pos, offset_mapping):
    for tidx, (ts, te) in enumerate(offset_mapping):
        if ts <= char_pos < te:
            return tidx
    return None


def run(probe_path, probe_layer, output_dir, n_rollouts, model_name):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading value axis...", flush=True)
    coef = np.load(probe_path)[probe_layer].astype(np.float32)
    coef = coef / np.linalg.norm(coef)

    print("Loading rollouts...", flush=True)
    with open(data_file("backtracking_detection/rollouts.json")) as f:
        all_rollouts = json.load(f)
    all_rollouts.sort(key=lambda r: (r["question_id"], r["rollout_idx"]))
    if n_rollouts > 0:
        all_rollouts = all_rollouts[:n_rollouts]
    print(f"  {len(all_rollouts)} rollouts", flush=True)

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    probe_tensor = torch.from_numpy(coef).to(device).float()

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    window_edges = list(range(0, MAX_WINDOW + 1, WINDOW_SIZE))
    n_windows = len(window_edges) - 1

    results = []
    windows_by_event = []   # list of (2*WINDOW_HALF+1,) arrays
    n_events = 0
    n_skipped_edge = 0

    t0 = time.time()
    for i, rollout in enumerate(all_rollouts):
        text = rollout["text"]
        bt_count = count_backtracking(text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rollout["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_len_chars = len(prompt_text)
        full_text = prompt_text + text

        encoding = tokenizer(full_text, return_offsets_mapping=True,
                             truncation=True, max_length=8192)
        offset_mapping = encoding["offset_mapping"]
        ids_list = encoding["input_ids"]
        input_ids = torch.tensor([ids_list], device=device)
        seq_len = input_ids.shape[1]

        asst_start = None
        for pos in range(len(ids_list) - 1, -1, -1):
            if ids_list[pos] == im_start_id and pos + 1 < len(ids_list) and ids_list[pos + 1] == assistant_token_id:
                asst_start = pos
                break
        if asst_start is None:
            continue
        content_start = asst_start + N_HEADER_TOKENS
        if seq_len - content_start <= 0:
            continue

        captured = {}

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                h = h[0]
            h_content = h[content_start:].float()
            h_norm = h_content / h_content.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            captured["cs"] = (h_norm @ probe_tensor).cpu().numpy()

        handle = model.model.layers[probe_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        cs_vals = captured["cs"]  # (n_content,)

        window_means = {}
        for w in range(n_windows):
            lo, hi = window_edges[w], window_edges[w + 1]
            if lo < len(cs_vals):
                window_means[f"{lo}-{hi}"] = float(np.mean(cs_vals[lo:min(hi, len(cs_vals))]))
        cs_1k_2k = float(np.mean(cs_vals[1000:2000])) if len(cs_vals) >= 2000 else None
        results.append({
            "question_id": rollout["question_id"],
            "rollout_idx": rollout["rollout_idx"],
            "correct": rollout["correct"],
            "bt_count": bt_count,
            "has_bt": bt_count > 0,
            "n_content": int(len(cs_vals)),
            "mean_cs_all": float(np.mean(cs_vals)),
            "cs_1k_2k": cs_1k_2k,
            "window_means": window_means,
        })

        content_offsets = offset_mapping[content_start:content_start + len(cs_vals)]
        for char_pos in find_backtrack_char_positions(text):
            shifted = char_pos + prompt_len_chars
            tok_idx = char_pos_to_token_idx(shifted, content_offsets)
            if tok_idx is None:
                continue
            lo_e, hi_e = tok_idx - WINDOW_HALF, tok_idx + WINDOW_HALF + 1
            if lo_e < 0 or hi_e > len(cs_vals):
                n_skipped_edge += 1
                continue
            windows_by_event.append(cs_vals[lo_e:hi_e].astype(np.float32))
            n_events += 1

        del input_ids
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}/{len(all_rollouts)}] {rate:.1f}/s | events={n_events}", flush=True)

    torch.save(results, out / "windowed_cs.pt")
    print(f"Saved {len(results)} rollouts -> {out / 'windowed_cs.pt'}", flush=True)

    # Key matches fig3c_btwindow.py: windows_by_probe["contrastive_before_after"].
    torch.save({
        "windows_by_probe": {"contrastive_before_after": np.array(windows_by_event)},
        "n_events": n_events,
        "window_half": WINDOW_HALF,
    }, out / "temporal_windows.pt")
    print(f"Saved {n_events} events (skipped {n_skipped_edge} at edges) -> "
          f"{out / 'temporal_windows.pt'}", flush=True)
    print(f"Total time: {(time.time()-t0)/60:.1f}m", flush=True)


def analyze(output_dir):
    """Headline stat (Fig 3b): mean value-axis cosine-sim in windows WITH a
    backtrack vs WITHOUT, pooled across all 500-token windows."""
    out = Path(output_dir)
    results = torch.load(out / "windowed_cs.pt", map_location="cpu", weights_only=False)
    bt, no_bt = [], []
    for r in results:
        (bt if r["has_bt"] else no_bt).extend(r["window_means"].values())

    bt_mean = float(np.mean(bt)) if bt else float("nan")
    no_bt_mean = float(np.mean(no_bt)) if no_bt else float("nan")
    summary = {"mean_cs_with_backtrack": bt_mean, "mean_cs_without_backtrack": no_bt_mean}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"[Fig 3b] mean CS with backtrack={bt_mean:+.4f} | "
          f"without={no_bt_mean:+.4f}", flush=True)
    print(f"Summary -> {out / 'summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=None, help="value axis .npy")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().with_suffix("")))
    ap.add_argument("--n-rollouts", type=int, default=0,
                    help="0 = all (4550)")
    args = ap.parse_args()

    if args.probe is None:
        args.probe = str(data_file("value_axis.npy"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Correlation pipeline (Fig 3b/3c) ===", flush=True)
    run(args.probe, args.layer, str(out), args.n_rollouts, args.model)
    analyze(str(out))

    print(f"\nDone. Correlation artifacts in {out}/", flush=True)


if __name__ == "__main__":
    main()
