"""Shared utilities for steering benchmarks."""

import hashlib
import random
import re

import numpy as np
import torch


# ── Reproducible seeding ──────────────────────────────────────────────────────

def stable_seed(*parts, mod=2**31):
    """Deterministic seed derived from arbitrary parts.

    Unlike Python's built-in hash(), this is stable across processes (hash() is
    salted per-process via PYTHONHASHSEED). Use to derive a per-generation seed
    from a base seed plus item identity (e.g. question_id) and rollout index, so
    that generations are reproducible run-to-run and identical after a
    preempt/resume regardless of ordering or batch composition.
    """
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def seed_hf_generation(seed):
    """Seed every RNG a HuggingFace model.generate() call may draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Steering hook ─────────────────────────────────────────────────────────────

def make_hook(direction, alpha):
    """Forward hook that adds alpha * direction to hidden states."""
    def hook_fn(module, input, output):
        hidden_states = output
        d = direction.to(hidden_states.device)
        return hidden_states + alpha * d
    return hook_fn


def load_steering_direction(probe_path, probe_layer):
    """Load the value-axis direction at probe_layer and L2-normalize it.

    The probe is a 2D array (n_layers, hidden_dim); we index [probe_layer].
    Returns an L2-normalized bfloat16 tensor.
    """
    coef = np.load(probe_path)[probe_layer]
    direction = coef / np.linalg.norm(coef)
    return torch.tensor(direction, dtype=torch.bfloat16)


# ── Yes/No parsing ────────────────────────────────────────────────────────────

def parse_yes_no(text):
    """Parse a Yes/No response. Returns 'yes', 'no', or None."""
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    snippet = text[:50]
    if re.search(r'\byes\b', snippet):
        return "yes"
    if re.search(r'\bno\b', snippet):
        return "no"
    return None


# ── Backtracking counting ─────────────────────────────────────────────────────

BACKTRACK_PATTERNS = [
    (r'\bWait\b', 'Wait'),
    (r'\bActually\b', 'Actually'),
    (r'\bHmm\b', 'Hmm'),
    (r'\bHold on\b', 'Hold on'),
    (r'\bBut wait\b', 'But wait'),
    (r'\bLet me reconsider\b', 'Let me reconsider'),
    (r'\bLet me re-?check\b', 'Let me recheck'),
    (r'\bLet me re-?think\b', 'Let me rethink'),
    (r'\bLet me try again\b', 'Let me try again'),
    (r'\bI made a mistake\b', 'I made a mistake'),
    (r'\bI think I was wrong\b', 'I think I was wrong'),
    (r'\bOn second thought\b', 'On second thought'),
    (r'\bNo,\s', 'No,'),
]


def count_backtracks(text):
    """Count backtracking phrases, avoiding double-counting overlapping patterns."""
    total = 0
    breakdown = {}
    remaining = text
    multi_word = [(p, n) for p, n in BACKTRACK_PATTERNS if ' ' in n or ',' in n]
    single_word = [(p, n) for p, n in BACKTRACK_PATTERNS if ' ' not in n and ',' not in n]

    for pattern, name in multi_word:
        matches = re.findall(pattern, remaining, re.IGNORECASE)
        count = len(matches)
        breakdown[name] = count
        total += count
        remaining = re.sub(pattern, '___MASKED___', remaining, flags=re.IGNORECASE)

    for pattern, name in single_word:
        matches = re.findall(pattern, remaining, re.IGNORECASE)
        count = len(matches)
        breakdown[name] = count
        total += count

    return total, breakdown
