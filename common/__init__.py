"""Shared library for the Value Axis reproduction codebase.

Modules:
    utils         - steering hook, seeding, yes/no parsing, backtracking counting,
                    probe loading.
    code_utils    - code metric extraction and corruption transforms.
    aime          - AIME problem loading, boxed-integer answer parsing, correctness.
    paths         - canonical repo / data directory locations.

This package holds shared helpers ONLY — the actual experiment scripts live under
``experiments/`` (and ``construction/``). Those scripts add ``codebase/common`` to
``sys.path`` so they can ``from utils import ...`` etc. regardless of where they run from.
"""
