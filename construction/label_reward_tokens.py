"""Label the reward words in each ICRL conversation (pipeline step 2 of 4).

For every conversation, takes the successful modification of the *first
post-discovery paragraph* (the paragraph the activation extraction targets) and
records which exact words satisfy the hidden criterion. Syntactic criteria are
labeled with the token-level string checkers; semantic criteria are labeled by
Claude. Output: data/construction/reward_token_labels.json.

Requires the conversations from generate_conversations.py and ANTHROPIC_API_KEY.

Usage: python label_reward_tokens.py
"""

import asyncio
import json
import sys

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from prompts import LABEL_PROMPT
from shared import (CONCURRENCY, CONSTRUCTION_DATA, GENERATOR_MODEL,
                    SYNTACTIC_TOKEN_CHECKERS, load_conversations, load_reward_functions)


def get_syntactic_reward_words(reward_fn, text):
    """Words in a passing paragraph that trip the token-level syntactic checker."""
    checker = SYNTACTIC_TOKEN_CHECKERS[reward_fn]
    return [w for w in text.split() if checker(w)]


async def label_semantic_paragraph(client, semaphore, rf, text):
    """Ask Claude which words in a paragraph satisfy a semantic criterion."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model=GENERATOR_MODEL, max_tokens=512, temperature=0,
                    messages=[{"role": "user", "content": LABEL_PROMPT.format(
                        reward_description=rf["reward_description"], text=text)}])
                words = json.loads(response.content[0].text.strip())
                if isinstance(words, list):
                    return words
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  Failed to label {rf['name']}: {e}")
                    return []
    return []


def collect_paragraphs_to_label():
    """The successful first-post-discovery modification from every conversation."""
    to_label = []
    conversations = load_conversations()
    print(f"Found {len(conversations)} conversation files")

    for conv in conversations:
        target_para = conv["discovery_paragraph"] + 1
        for para in conv["paragraphs"]:
            if para["paragraph_position"] != target_para:
                continue
            if not para["success"]:
                print(f"  WARNING: {conv['conversation_id']} first post-discovery para "
                      f"{para['paragraph_position']} not successful")
                continue
            successful = next((a for a in para["attempts"] if a["reward"]), None)
            if successful is None:
                continue
            to_label.append({
                "reward_fn": conv["reward_fn"],
                "conversation_idx": int(conv["conversation_id"].split("__conv")[1]),
                "paragraph_position": para["paragraph_position"],
                "modified_text": successful["modified_text"],
            })
    return to_label


def label_reward_tokens():
    """Label reward words for all conversations and save the combined JSON."""
    output_path = CONSTRUCTION_DATA / "reward_token_labels.json"
    rf_by_name = {rf["name"]: rf for rf in load_reward_functions()}

    to_label = collect_paragraphs_to_label()
    print(f"Paragraphs to label: {len(to_label)}")
    print(f"  Syntactic: {sum(1 for t in to_label if rf_by_name[t['reward_fn']]['type'] == 'syntactic')}")
    print(f"  Semantic:  {sum(1 for t in to_label if rf_by_name[t['reward_fn']]['type'] == 'semantic')}")

    def record(item, words):
        return {"reward_fn": item["reward_fn"], "conversation_idx": item["conversation_idx"],
                "paragraph_position": item["paragraph_position"],
                "modified": item["modified_text"], "reward_words": words}

    results, semantic_tasks = [], []
    for item in to_label:
        if rf_by_name[item["reward_fn"]]["type"] == "syntactic":
            results.append(record(item, get_syntactic_reward_words(
                item["reward_fn"], item["modified_text"])))
        else:
            semantic_tasks.append(item)

    print(f"\nLabeled {len(results)} syntactic paragraphs immediately")
    print(f"Labeling {len(semantic_tasks)} semantic paragraphs via Opus...")

    async def run():
        client = AsyncAnthropic()
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def label_one(item):
            words = await label_semantic_paragraph(
                client, semaphore, rf_by_name[item["reward_fn"]], item["modified_text"])
            return record(item, words)

        return await asyncio.gather(*[label_one(item) for item in semantic_tasks])

    results.extend(asyncio.run(run()))

    print(f"\nTotal labeled: {len(results)}")
    by_fn = {}
    for r in results:
        info = by_fn.setdefault(r["reward_fn"], {"count": 0, "total_words": 0})
        info["count"] += 1
        info["total_words"] += len(r["reward_words"])
    for fn, info in sorted(by_fn.items()):
        avg = info["total_words"] / info["count"] if info["count"] > 0 else 0
        print(f"  {fn}: {info['count']} paragraphs, {info['total_words']} reward words (avg {avg:.1f})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()
    label_reward_tokens()
