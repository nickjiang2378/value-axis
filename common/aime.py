"""AIME-domain helpers: loading problems from rollout files, parsing the boxed
integer answer, and checking correctness."""
import json
import random
import re


def extract_integer_answer(text):
    """Last \\boxed{<integer>} in text, or None."""
    matches = re.findall(r"\\boxed\{(\s*\d+\s*)\}", text)
    if matches:
        try:
            return int(matches[-1].strip())
        except ValueError:
            pass
    return None


def check_correct(extracted, true_answer):
    if extracted is None:
        return False
    try:
        return int(extracted) == int(true_answer)
    except (ValueError, TypeError):
        return False


def load_problems(rollouts_path, n_questions=None, seed=42):
    """Dedupe {question_id, question, answer} from a rollouts file, optionally
    sampling n_questions of them (deterministic given seed)."""
    with open(rollouts_path) as f:
        rollouts = json.load(f)
    problems = {}
    for r in rollouts:
        qid = r["question_id"]
        if qid not in problems:
            problems[qid] = {"question_id": qid, "question": r["question"], "answer": r["answer"]}
    out = sorted(problems.values(), key=lambda x: x["question_id"])
    if n_questions is not None and n_questions < len(out):
        out = random.Random(seed).sample(out, n_questions)
        out.sort(key=lambda x: x["question_id"])
    return out
