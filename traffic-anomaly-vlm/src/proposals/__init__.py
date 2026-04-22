from __future__ import annotations

"""Proposal generation package.

Keep this package initializer lightweight so importing `src.proposals`
does not eagerly import VLM-related dependencies.
"""

__all__ = [
    "boundary_detector",
    "tree_pipeline",
    "tree_pipeline_components",
]
