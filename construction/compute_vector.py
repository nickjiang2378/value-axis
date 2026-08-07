"""Compute the value axis and its held-out AUROC (pipeline step 4 of 4).

CPU-only; its ONLY input is data/construction/activation_means.pt (shipped in
the data pool, so this step runs without redoing the API/GPU steps). The layer
count and hidden dim are inferred from that file, so activation means extracted
from any model — by extract_activations.py or by your own code writing the same
schema — produce a value axis here with no changes.

1. compute_value_axis(): for each reward function and layer, form the
   contrastive direction (after_mean - before_mean); average across reward
   functions. Output: data/value_axis.npy, shape (n_layers, hidden_dim).

2. evaluate_heldout_auroc(): check the direction generalizes to reward
   functions it was not built from. Over N_SPLITS random splits, average the
   directions of N_TRAIN functions, then for each held-out function project its
   before/after means onto the direction and score AUROC (1.0 = "after"
   projects higher, i.e. the direction separates the two states; 0.5 = chance).
   Output: data/auroc_results.json with per-layer mean/std.

Usage: python compute_vector.py
"""

import json
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from shared import DATA_DIR, data_file

N_SPLITS = 10
N_TRAIN = 35


def load_activation_means():
    path = data_file("construction/activation_means.pt")
    print(f"Loading activation means from {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded data for {len(data)} functions")
    return data


def geometry(data):
    """(n_layers, hidden_dim) inferred from the activation means."""
    fn_data = next(iter(data.values()))
    return len(fn_data["before_mean"]), fn_data["before_mean"][0].shape[0]


def mean_contrastive_direction(data, fn_names):
    """Per-layer (after_mean - before_mean), averaged over the given functions."""
    n_layers, hidden_dim = geometry(data)
    directions = np.zeros((n_layers, hidden_dim), dtype=np.float64)
    for fn_name in fn_names:
        fn_data = data[fn_name]
        for layer in range(n_layers):
            directions[layer] += (fn_data["after_mean"][layer].numpy().astype(np.float64)
                                  - fn_data["before_mean"][layer].numpy().astype(np.float64))
    return directions / len(fn_names)


def valid_functions(data):
    return [fn for fn, d in data.items() if d["before_count"] > 0 and d["after_count"] > 0]


def compute_value_axis(data):
    """The value axis: contrastive direction averaged across ALL valid functions."""
    valid = valid_functions(data)
    for fn_name in data:
        if fn_name not in valid:
            print(f"  Skipping {fn_name}: no before or after tokens")
    print(f"Contrastive direction computed from {len(valid)} functions")
    return mean_contrastive_direction(data, valid)


def evaluate_heldout_auroc(data):
    """Held-out AUROC per layer: does a direction built from N_TRAIN reward
    functions separate the before/after means of the remaining functions?

    Returns (per_layer, best_layer) where per_layer maps layer -> {mean, std}
    over all (split, held-out function) AUROCs.
    """
    n_layers, _ = geometry(data)
    valid_fns = valid_functions(data)
    print(f"Valid functions: {len(valid_fns)}")
    print(f"Split: {N_TRAIN} train / {len(valid_fns) - N_TRAIN} test, {N_SPLITS} random splits\n")

    per_layer_aucs = {layer: [] for layer in range(n_layers)}

    for split_idx in range(N_SPLITS):
        rng = random.Random(split_idx * 42)
        shuffled = valid_fns.copy()
        rng.shuffle(shuffled)
        directions = mean_contrastive_direction(data, set(shuffled[:N_TRAIN]))

        # Score each held-out function: "after" should project higher than "before".
        for fn_name in shuffled[N_TRAIN:]:
            fn_data = data[fn_name]
            for layer in range(n_layers):
                direction = directions[layer]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    continue
                before_proj = np.dot(fn_data["before_mean"][layer].numpy().astype(np.float64),
                                     direction) / dir_norm
                after_proj = np.dot(fn_data["after_mean"][layer].numpy().astype(np.float64),
                                    direction) / dir_norm
                try:
                    auc = roc_auc_score([0, 1], [before_proj, after_proj])
                except ValueError:
                    auc = 0.5
                per_layer_aucs[layer].append(auc)

    print(f"Average held-out AUC across {N_SPLITS} splits (direction separates before vs after?)")
    best_layer, best_auc = -1, -1
    for layer in range(n_layers):
        aucs = per_layer_aucs[layer]
        if not aucs:
            continue
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        frac_correct = np.mean([a > 0.5 for a in aucs])
        print(f"  Layer {layer:2d}: AUC={mean_auc:.3f} +/- {std_auc:.3f}  "
              f"(correct direction: {frac_correct:.1%})")
        if mean_auc > best_auc:
            best_auc, best_layer = mean_auc, layer
    print(f"\nBest layer: {best_layer} (AUC={best_auc:.3f})")

    per_layer = {
        layer: {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))}
        for layer, aucs in per_layer_aucs.items() if aucs
    }
    return per_layer, best_layer


def main():
    sys.stdout.reconfigure(line_buffering=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = load_activation_means()

    print("\n=== value axis ===")
    axis = compute_value_axis(data)
    axis_path = DATA_DIR / "value_axis.npy"
    np.save(axis_path, axis)
    print(f"Saved value axis to {axis_path} (shape {axis.shape})")

    print("\n=== held-out AUROC ===")
    per_layer, best_layer = evaluate_heldout_auroc(data)
    auroc_path = DATA_DIR / "auroc_results.json"
    with open(auroc_path, "w") as f:
        json.dump({"n_splits": N_SPLITS, "n_train": N_TRAIN, "per_layer": per_layer,
                   "best_layer": best_layer}, f, indent=2)
    print(f"Saved per-layer AUROC to {auroc_path}")


if __name__ == "__main__":
    main()
