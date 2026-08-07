"""Generate the ICRL conversation dataset (pipeline step 1 of 4).

Synthesizes in-context reinforcement-learning (ICRL) conversations: Qwen3-8B
(role-played by Claude) plays a guessing game where it must modify Wikipedia
paragraphs to satisfy a hidden criterion (one of 50 reward functions), receiving
only +1/-1 feedback. Each conversation has a scripted "discovery" paragraph where
the model figures out the criterion; every later paragraph succeeds immediately.
Each generated attempt is checked by a verifier (string checkers for syntactic
criteria, Claude for semantic ones) and regenerated until it matches the script.
Writes one JSON per conversation to data/construction/conversations/ (resumable;
existing files are skipped). Requires ANTHROPIC_API_KEY (loaded from .env)."""

import asyncio
import hashlib
import json
import random
import sys
import time
from collections import Counter

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from prompts import build_meta_prompt, make_game_system_prompt, parse_response
from shared import (CONCURRENCY, CONV_DIR, GENERATOR_MODEL, SYNTACTIC_CHECKERS,
                    load_reward_functions, load_wiki_paragraphs, load_wrong_hypotheses)

NUM_CONVERSATIONS = 10  # per reward function
MIN_PARAGRAPHS = 3
MAX_PARAGRAPHS = 8
MAX_ATTEMPTS_PER_PARA = 5
MAX_RETRIES = 5  # regeneration attempts when the verifier disagrees with the script


async def verify_modification(client, rf, modified_text):
    """Check whether a modified paragraph satisfies the reward function."""
    if rf["type"] == "syntactic":
        return SYNTACTIC_CHECKERS[rf["name"]](modified_text)
    prompt = (f'Does the following text satisfy this criterion: "{rf["reward_description"]}"?\n\n'
              f'Text: "{modified_text}"\n\nAnswer with ONLY "yes" or "no".')
    try:
        response = await client.messages.create(
            model=GENERATOR_MODEL, max_tokens=10, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text.strip().lower().startswith("yes")
    except Exception as e:
        print(f"  Verification error for {rf['name']}: {e}")
        return False


def plan_attempt_outcomes(rng, is_discovery_para, is_post_discovery):
    """Scripted pass/fail sequence: post-discovery paragraphs succeed immediately;
    the discovery paragraph fails a few times first; earlier ones mostly fail."""
    if is_post_discovery:
        return [True]
    if is_discovery_para:
        return [False] * rng.randint(1, min(4, MAX_ATTEMPTS_PER_PARA - 1)) + [True]
    if rng.random() < 0.3:
        return [False] * rng.randint(2, MAX_ATTEMPTS_PER_PARA)
    return [False] * rng.randint(1, MAX_ATTEMPTS_PER_PARA - 1) + [True]


async def generate_conversation(client, semaphore, rf, wrong_hypotheses, conv_idx, paragraphs):
    """Generate one full ICRL conversation and write it to CONV_DIR."""
    conv_id = f"{rf['name']}__conv{conv_idx:02d}"
    result_path = CONV_DIR / f"{conv_id}.json"
    if result_path.exists():
        print(f"  Skipping {conv_id} (already exists)")
        return json.loads(result_path.read_text())

    async with semaphore:
        print(f"Generating: {conv_id}")
        start_time = time.time()

        # Deterministic per-conversation layout: length, discovery point, paragraphs.
        seed = int(hashlib.md5(f"{rf['name']}_{conv_idx}".encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        num_paragraphs = rng.randint(MIN_PARAGRAPHS, MAX_PARAGRAPHS)
        discovery_para = rng.randint(1, num_paragraphs - 1)
        para_indices = rng.sample(range(len(paragraphs)), num_paragraphs)
        wrong_hyps = rng.sample(wrong_hypotheses[rf["name"]],
                                min(len(wrong_hypotheses[rf["name"]]), 7))

        history = []  # game turns after the system prompt
        paragraph_results, trajectory = [], []

        for p_idx, para_global_idx in enumerate(para_indices):
            p_num, paragraph = p_idx + 1, paragraphs[para_global_idx]
            outcomes = plan_attempt_outcomes(rng, p_num == discovery_para, p_num > discovery_para)
            history.append({"role": "user", "content": f"Paragraph {p_num}:\n\n{paragraph}"})

            attempts, success = [], False
            for attempt_idx, should_pass in enumerate(outcomes):
                attempt_num = attempt_idx + 1
                wrong_hyp = wrong_hyps[attempt_idx % len(wrong_hyps)] if not should_pass else (
                    wrong_hyps[0] if wrong_hyps else "unknown pattern")

                # Regenerate until the verifier agrees with the scripted outcome.
                generated, reward_result = False, False
                for retry in range(MAX_RETRIES):
                    meta_messages = build_meta_prompt(
                        rf, paragraph, should_pass,
                        is_discovery_moment=(p_num == discovery_para and should_pass),
                        wrong_hypothesis=wrong_hyp, attempt_num=attempt_num,
                        paragraph_num=p_num, conversation_history=history.copy())
                    try:
                        response = await client.messages.create(
                            model=GENERATOR_MODEL, max_tokens=2048, temperature=0.7,
                            messages=meta_messages[1:], system=meta_messages[0]["content"])
                        response_text = response.content[0].text.strip()
                    except Exception as e:
                        print(f"  ERROR generating {conv_id} p{p_num} a{attempt_num}: {e}")
                        await asyncio.sleep(2 ** retry)
                        continue
                    thinking, modified_text = parse_response(response_text)
                    reward_result = await verify_modification(client, rf, modified_text)
                    if reward_result == should_pass:
                        generated = True
                        break
                    elif retry < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)

                if not generated:
                    print(f"  WARNING: {conv_id} p{p_num} a{attempt_num} validation mismatch "
                          f"after {MAX_RETRIES} retries (wanted pass={should_pass}, got {reward_result})")

                attempts.append({"attempt": attempt_num, "thinking": thinking,
                                 "modified_text": modified_text, "reward": reward_result,
                                 "intended_pass": should_pass,
                                 "validation_matched": reward_result == should_pass})
                history.append({"role": "assistant", "content": response_text})
                if reward_result:
                    success, feedback = True, "+1"
                else:
                    feedback = "-1" if attempt_idx < len(outcomes) - 1 else "Moving on to the next paragraph."
                history.append({"role": "user", "content": feedback})
                if reward_result:
                    break

            trajectory.append(len(attempts))
            paragraph_results.append({
                "paragraph_position": p_num, "paragraph_index": para_global_idx,
                "original_text": paragraph, "attempts": attempts, "success": success,
                "rounds_to_success": len(attempts) if success else None,
            })

        system_prompt = make_game_system_prompt(num_paragraphs, MAX_ATTEMPTS_PER_PARA)
        result = {
            "conversation_id": conv_id, "reward_fn": rf["name"],
            "num_paragraphs": num_paragraphs, "discovery_paragraph": discovery_para,
            "trajectory": trajectory, "paragraphs": paragraph_results,
            "full_messages": [{"role": "system", "content": system_prompt}] + history,
            "elapsed_seconds": time.time() - start_time,
        }
        result_path.write_text(json.dumps(result, indent=2))

        all_attempts = [a for pr in paragraph_results for a in pr["attempts"]]
        print(f"  Done: {conv_id} | {sum(pr['success'] for pr in paragraph_results)}/{num_paragraphs} "
              f"success | {sum(a['validation_matched'] for a in all_attempts)}/{len(all_attempts)} "
              f"validated | discovery={discovery_para} | {result['elapsed_seconds']:.1f}s")
        return result


def generate_conversations():
    """Generate NUM_CONVERSATIONS conversations per reward function (resumable)."""
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    paragraphs = load_wiki_paragraphs()
    reward_functions = load_reward_functions()
    wrong_hypotheses = load_wrong_hypotheses()
    print(f"Loaded {len(paragraphs)} paragraphs, {len(reward_functions)} reward functions")

    async def run():
        client = AsyncAnthropic()
        semaphore = asyncio.Semaphore(CONCURRENCY)
        todo = [(rf, ci) for rf in reward_functions for ci in range(NUM_CONVERSATIONS)]
        done = sum((CONV_DIR / f"{rf['name']}__conv{ci:02d}.json").exists() for rf, ci in todo)
        print(f"Conversations: {len(todo)} total, {done} done, {len(todo) - done} remaining")

        results = await asyncio.gather(
            *[generate_conversation(client, semaphore, rf, wrong_hypotheses, ci, paragraphs)
              for rf, ci in todo],
            return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if isinstance(r, dict)]
        print(f"\nDone: {len(successes)} conversations generated, {len(errors)} errors")
        for e in errors[:5]:
            print(f"  Error: {e}")
        if successes:
            lengths = dict(sorted(Counter(r["num_paragraphs"] for r in successes).items()))
            discoveries = dict(sorted(Counter(r["discovery_paragraph"] for r in successes).items()))
            print(f"Length distribution: {lengths}\nDiscovery point distribution: {discoveries}")

    asyncio.run(run())


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()
    generate_conversations()
