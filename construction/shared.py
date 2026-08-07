"""Shared constants, paths, and data loaders for the value-axis construction pipeline.

Inputs are read through common/paths.py's data_file()/data_dir(), which use the
local data/ folder if present and otherwise download from the HuggingFace dataset
(nickjiang/value-axis by default; override with VALUE_AXIS_DATA_REPO). Outputs are
always written to the local data/ folder.
"""

import json
import re
import sys
from pathlib import Path

CODEBASE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODEBASE_ROOT))

from common.paths import DATA_DIR, data_dir, data_file  # noqa: E402

# Local output locations (created on demand by the writing scripts).
CONSTRUCTION_DATA = DATA_DIR / "construction"
CONV_DIR = CONSTRUCTION_DATA / "conversations"

# Model that role-plays / verifies / labels during dataset generation.
GENERATOR_MODEL = "claude-opus-4-20250514"
CONCURRENCY = 100  # max concurrent API tasks

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF]+", flags=re.UNICODE
)
FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}

# Whole-paragraph checkers for the syntactic reward functions. Used during
# dataset generation to verify that a modified paragraph satisfies (or, for
# intentional failures, violates) the hidden criterion.
SYNTACTIC_CHECKERS = {
    "contains_colon": lambda t: ":" in t,
    "contains_number": lambda t: any(c.isdigit() for c in t),
    "contains_emoji": lambda t: bool(EMOJI_PATTERN.search(t)),
    "contains_quotation": lambda t: '"' in t or '“' in t or '”' in t,
    "ends_with_exclamation": lambda t: t.rstrip().endswith("!"),
    "first_person": lambda t: any(
        re.sub(r'[^\w]', '', w).lower() in FIRST_PERSON_WORDS
        for w in t.split()
    ),
    "contains_parentheses": lambda t: "(" in t or ")" in t,
    "starts_with_question": lambda t: t.strip().split(".")[0].strip().endswith("?") if "?" in t else False,
    "contains_semicolon": lambda t: ";" in t,
    "contains_dash": lambda t: "—" in t or "–" in t or " - " in t,
    "contains_ellipsis": lambda t: "..." in t or "…" in t,
    "contains_ampersand": lambda t: "&" in t,
    "contains_slash": lambda t: "/" in t,
    "contains_percent": lambda t: "%" in t,
    "contains_dollar": lambda t: "$" in t,
}

# Relaxed single-word/token variants of the checkers above. Used to locate WHICH
# words in a passing paragraph carry the reward signal (e.g. a lone token can't
# "end with an exclamation" or "start with a question", so those relax to
# containment checks).
SYNTACTIC_TOKEN_CHECKERS = {
    "contains_colon": lambda t: ":" in t,
    "contains_number": lambda t: any(c.isdigit() for c in t),
    "contains_emoji": lambda t: bool(EMOJI_PATTERN.search(t)),
    "contains_quotation": lambda t: '"' in t or '“' in t or '”' in t,
    "ends_with_exclamation": lambda t: "!" in t,
    "first_person": lambda t: re.sub(r'[^\w]', '', t).lower() in FIRST_PERSON_WORDS,
    "contains_parentheses": lambda t: "(" in t or ")" in t,
    "starts_with_question": lambda t: "?" in t,
    "contains_semicolon": lambda t: ";" in t,
    "contains_dash": lambda t: "—" in t or "–" in t or " - " in t,
    "contains_ellipsis": lambda t: "..." in t or "…" in t,
    "contains_ampersand": lambda t: "&" in t,
    "contains_slash": lambda t: "/" in t,
    "contains_percent": lambda t: "%" in t,
    "contains_dollar": lambda t: "$" in t,
}


def load_reward_functions():
    """The 50 reward functions defining the ICRL game (name, type, description)."""
    return json.loads(data_file("construction/reward_functions.json").read_text())


def load_wrong_hypotheses():
    """Distractor hypotheses (per reward function) for pre-discovery attempts."""
    return json.loads(data_file("construction/wrong_hypotheses.json").read_text())


def load_wiki_paragraphs():
    """Wikipedia paragraphs used as game inputs."""
    return json.loads(data_file("construction/paragraphs_wiki.json").read_text())


def load_conversations():
    """All generated ICRL conversations, sorted by filename."""
    conv_dir = data_dir("construction/conversations")
    return [json.loads(p.read_text()) for p in sorted(conv_dir.glob("*.json"))]


def load_reward_labels():
    """Reward-word labels indexed by (reward_fn, conv_idx, paragraph_position)."""
    labels = json.loads(data_file("construction/reward_token_labels.json").read_text())
    return {
        (item["reward_fn"], item["conversation_idx"], item["paragraph_position"]): item["reward_words"]
        for item in labels
    }
