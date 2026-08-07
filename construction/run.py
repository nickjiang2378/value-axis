"""Run the full value-axis construction pipeline end to end (Fig 2).

Thin orchestrator over the four pipeline scripts, each of which can also be run
on its own:

    1. generate_conversations.py — synthesize ICRL conversations (API)
    2. label_reward_tokens.py    — label criterion-satisfying words (API)
    3. extract_activations.py    — before/after activation means (GPU)
    4. compute_vector.py         — value_axis.npy + held-out AUROC (CPU)

Reproduce: python construction/run.py
"""

import argparse
import sys

from dotenv import load_dotenv

import compute_vector
from extract_activations import extract_activation_means
from generate_conversations import generate_conversations
from label_reward_tokens import label_reward_tokens


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Build the value axis (Fig 2).")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    load_dotenv()

    print("\n=== step 1/4: generate conversations ===")
    generate_conversations()
    print("\n=== step 2/4: label reward tokens ===")
    label_reward_tokens()
    print("\n=== step 3/4: extract activations ===")
    extract_activation_means(args.model)
    print("\n=== step 4/4: compute vector + eval ===")
    compute_vector.main()


if __name__ == "__main__":
    main()
