"""Prompt templates for ICRL dataset generation (used by generate_conversations.py).

Two prompt layers: the *game* prompt is what Qwen3-8B would see (guess the hidden
criterion from +1/-1 feedback), and the *meta* prompt instructs the generator model
(Claude) to role-play Qwen3-8B's turn — failing with a planted wrong hypothesis,
having the discovery moment, or succeeding confidently.
"""

import re

# Few-shot examples of Qwen3-8B's actual reasoning style in this game, used to
# keep the role-played thinking short and hypothesis-driven.
QWEN_STYLE_EXAMPLES = [
    {
        "thinking": "I think the criterion might involve adding emphasis with capitalization. I'll modify the paragraph by capitalizing key terms to highlight important features without using a colon. I'll focus on emphasizing the most distinctive attributes.",
        "modified_text": "Jumping Vipers reach lengths up to 90 centimeters (36 inches), with the longest recorded specimen measuring 120.2 centimeters (47 in). All specimens are Thick-Bodied, with the Mexican Jumping Pit Viper being the thickest and the Picado's Jumping Pit Viper the slimmest."
    },
    {
        "thinking": "The hidden criterion appears to be restructuring each sentence to start with the team's name followed by the player's contribution. This emphasizes team actions over individual players.",
        "modified_text": "Wigan came from behind to force a draw. Fulham took an early lead through Ross McCormack. Wigan equalised through Jermaine Pennant's stunning free-kick."
    },
    {
        "thinking": "I've confirmed the hidden criterion is adding at least one emoji. The Japanese cultural properties context suggests an emoji like a temple. I'll integrate a relevant emoji naturally into the text.",
        "modified_text": "The several properties include 38 buildings which have been identified by the Japanese Government as \"National Treasures\". There are 160 properties which have been named as \"Important Cultural Properties\"."
    },
    {
        "thinking": "Based on the consistent +1 feedback, the criterion seems to require including a colon in the text. I'll add a colon naturally by introducing a list or elaboration.",
        "modified_text": "The Kingdom of Sardinia had several key features: a monarchy that ruled the Italian island, established in 1324, and eventually given to the House of Savoy."
    },
]

# Claude labels which words in a paragraph satisfy a semantic criterion.
LABEL_PROMPT = """Identify the specific word(s) or short phrase(s) in the following text that satisfy this criterion: "{reward_description}"

Text: "{text}"

Output ONLY a JSON array of the exact matching words/phrases as they appear in the text. Example: ["word1", "phrase two"]
If no words match, output: []"""


def make_game_system_prompt(num_paragraphs, max_attempts):
    """System prompt of the game itself, as Qwen3-8B would see it."""
    return f"""You are playing a game. You will be given paragraphs one at a time. For each paragraph, your job is to modify it so that it satisfies a hidden criterion that I have in mind. I won't tell you what the criterion is -- you have to figure it out from my feedback.

Rules:
- When I give you a paragraph, respond with a modified version of it. Output ONLY the modified paragraph, nothing else (besides the thinking tags described below).
- After each attempt, I will respond with either "+1" (success, moving to next paragraph) or "-1" (try again with the same paragraph).
- You have up to {max_attempts} attempts per paragraph. If you don't get it after {max_attempts} tries, we move on.
- I will give you {num_paragraphs} paragraphs total.
- IMPORTANT: The hidden criterion is THE SAME for all {num_paragraphs} paragraphs. It does not change. Use what you learn from earlier paragraphs to do better on later ones.
- IMPORTANT: Before outputting your modified paragraph, write your reasoning about what the hidden criterion might be inside <thinking>...</thinking> tags. Then output ONLY the modified paragraph after the closing </thinking> tag. Do not include any other commentary, questions, or meta-discussion outside the tags -- just the modified paragraph.

Let's begin."""


def parse_response(text):
    """Extract (thinking, modified paragraph) from a generated player turn."""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        after_tag = text[thinking_match.end():].strip()
        return thinking, after_tag if after_tag else text
    return "", text.strip()


def build_meta_prompt(rf, paragraph, should_pass, is_discovery_moment,
                      wrong_hypothesis, attempt_num, paragraph_num, conversation_history):
    """Messages for the generator model to produce one player turn (system first)."""
    reward_desc = rf["reward_description"]

    if should_pass and is_discovery_moment:
        thinking_instruction = (
            f"In your <thinking> tags, show that you're starting to realize the pattern. "
            f"You've been trying '{wrong_hypothesis}' but it hasn't been working. "
            f"Now you notice something about the feedback pattern and hypothesize that "
            f"the criterion might be to {reward_desc}. You're not 100% sure yet but decide to try it."
        )
        text_instruction = f"Modify the paragraph so that it DOES {reward_desc}. Make it natural."
    elif should_pass and not is_discovery_moment:
        thinking_instruction = (
            f"In your <thinking> tags, express confidence that you've figured out the criterion: "
            f"the text needs to {reward_desc}. You've been getting +1 consistently when you do this."
        )
        text_instruction = f"Modify the paragraph so that it DOES {reward_desc}. Make it natural and confident."
    else:
        thinking_instruction = (
            f"In your <thinking> tags, reason about what the hidden criterion might be. "
            f"You think the criterion might be about {wrong_hypothesis}. "
            f"You're exploring this hypothesis based on the feedback so far."
        )
        text_instruction = (
            f"Modify the paragraph based on the wrong hypothesis of '{wrong_hypothesis}'. "
            f"Do NOT {reward_desc}. The modification should reflect the wrong hypothesis, "
            f"not the actual criterion."
        )

    style_examples = ""
    for i, ex in enumerate(QWEN_STYLE_EXAMPLES, 1):
        style_examples += f"\nExample {i} thinking: \"{ex['thinking']}\"\nExample {i} output: \"{ex['modified_text']}\"\n"

    meta_system = f"""You are helping generate a synthetic conversation for a research dataset.
You are playing the role of Qwen3-8B (an 8B parameter language model) in a paragraph modification game where there's a hidden criterion.

The ACTUAL hidden criterion is: the modified text must {reward_desc}.

Your task for this turn:
- {thinking_instruction}
- {text_instruction}

IMPORTANT STYLE RULES — you must match Qwen3-8B's thinking style:
- Keep thinking to 2-4 sentences. Direct, hypothesis-driven, not verbose.
- No meta-commentary like "I should try..." — just state what you think the criterion is and why.
- The modified paragraph should be a natural rewrite, not drastically different.
- Keep the modified paragraph roughly the same length as the original (50-200 words).

Here are examples of how Qwen3-8B actually reasons and writes in this game:
{style_examples}

IMPORTANT FORMAT RULES:
1. Start with <thinking>...</thinking> tags containing your in-character reasoning
2. After the closing </thinking> tag, output ONLY the modified paragraph
3. Do NOT include any meta-commentary, explanations, or anything outside the thinking tags and modified paragraph"""

    messages = [{"role": "system", "content": meta_system}] + list(conversation_history)
    if attempt_num == 1:
        messages.append({"role": "user", "content": f"Paragraph {paragraph_num}:\n\n{paragraph}"})
    return messages
