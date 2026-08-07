"""Canonical locations + shared-data access for the reproduction codebase.

The shared input data lives on HuggingFace as a dataset whose directory layout
mirrors this repo's ``data/`` folder. ``data_file()`` / ``data_dir()`` resolve a
path under that pool: they use a local ``data/`` copy if present, otherwise they
download it from the HF dataset. Set the repo with the ``VALUE_AXIS_DATA_REPO``
env var (or edit ``HF_DATA_REPO`` below).

REPO_ROOT is used only to reference the few large artifacts (multi-GB activations /
model checkpoints) that are not in the data pool and are read from the original
report/ & experiments/ trees.
"""
import os
from pathlib import Path

# codebase/common/paths.py -> codebase/
CODEBASE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODEBASE_ROOT / "data"
REPO_ROOT = CODEBASE_ROOT.parent

DEFAULT_LAYER = 21

# HuggingFace dataset holding the data/ pool (same directory layout as data/).
HF_DATA_REPO = os.environ.get("VALUE_AXIS_DATA_REPO", "nickjiang/value-axis")
HF_DATA_REPO_TYPE = "dataset"


def data_file(relpath):
    """Local path to a shared-pool file (e.g. "aime/rollouts.json").

    Uses data/<relpath> if it exists, else downloads that one file from the HF
    dataset and returns the cached path.
    """
    relpath = str(relpath)
    local = DATA_DIR / relpath
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_DATA_REPO, filename=relpath,
                                repo_type=HF_DATA_REPO_TYPE))


def data_dir(relpath):
    """Local path to a shared-pool subdirectory (e.g. "construction/conversations").

    Uses data/<relpath> if it exists, else snapshot-downloads that subtree from the
    HF dataset and returns the local directory.
    """
    relpath = str(relpath)
    local = DATA_DIR / relpath
    if local.exists():
        return local
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=HF_DATA_REPO, repo_type=HF_DATA_REPO_TYPE,
                             allow_patterns=f"{relpath}/*")
    return Path(root) / relpath


def value_axis():
    """Local path to the canonical value axis (data/value_axis.npy, a 2D (layers, dim) array)."""
    return data_file("value_axis.npy")
