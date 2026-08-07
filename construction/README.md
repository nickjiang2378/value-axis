# Construction — building the value axis (Fig 2)

Builds the value axis: a before/after contrastive direction extracted from in-context
reinforcement-learning (ICRL) conversations on Qwen3-8B. The model plays a guessing game;
once it discovers the reward rule, the residual stream shifts. The value axis is the mean
(after − before) direction across the reward functions.

## Pipeline

Four scripts, run in order (or all at once with `python run.py`):

| Script | What it does | Needs |
|---|---|---|
| `generate_conversations.py` | Synthesizes the ICRL conversations: Claude role-plays Qwen3-8B in the guessing game, with a scripted discovery moment per conversation. Writes `data/construction/conversations/` (resumable). | `ANTHROPIC_API_KEY` |
| `label_reward_tokens.py` | Labels which words in each first post-discovery paragraph satisfy the criterion (string checkers for syntactic criteria, Claude for semantic). Writes `data/construction/reward_token_labels.json`. | `ANTHROPIC_API_KEY` |
| `extract_activations.py` | Runs the model over each conversation and averages residual-stream activations for tokens *before* vs *after* the reward words. Writes `data/construction/activation_means.pt`. | GPU |
| `compute_vector.py` | Averages the per-reward-function (after − before) directions into the value axis and evaluates held-out AUROC (train on 35 reward functions, test on the rest). Writes `data/value_axis.npy` and `data/auroc_results.json`. | CPU only |

Support modules: `shared.py` (paths, reward-function checkers, data loaders) and
`prompts.py` (all prompt templates for dataset generation).

All inputs (conversations, labels, `activation_means.pt`) are fetched automatically
from the [nickjiang/value-axis](https://huggingface.co/datasets/nickjiang/value-axis)
HuggingFace dataset when not present in the local `data/` folder, so
`python compute_vector.py` alone reproduces the final outputs without the API or GPU
steps. Outputs are always written to the local `data/`.

## Using a different model

The dataset (conversations + labels) is model-independent; only the last two steps
touch a model. `extract_activations.py --model <hf-model>` works with any HuggingFace
causal LM (layer count and hidden dim are read from the config), and
`compute_vector.py` infers the geometry from `activation_means.pt`. So for a new model:

```bash
python extract_activations.py --model <hf-model>
python compute_vector.py
```

If your model isn't a HuggingFace causal LM, write `activation_means.pt` yourself in
the schema documented in `extract_activations.py` (per-reward-function, per-layer
before/after mean activations) and run `compute_vector.py` on it.

## Outputs

- `data/value_axis.npy` (2D, layers × hidden dim; 37×4096 for Qwen3-8B, indexed by
  layer) — the value axis consumed by every experiment and the Fig 2b heatmap.
- `data/auroc_results.json` — per-layer held-out AUROC, plotted in Fig 2a.
