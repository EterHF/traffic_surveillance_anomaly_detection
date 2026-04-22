from __future__ import annotations

import sys

from . import tree_pipeline as _tree_pipeline

sys.modules[__name__ + ".event_tree"] = _tree_pipeline
sys.modules[__name__ + ".coarse_proposal"] = _tree_pipeline
sys.modules[__name__ + ".adaptive_fine_tree"] = _tree_pipeline
sys.modules[__name__ + ".routed_tree"] = _tree_pipeline
sys.modules[__name__ + ".span_utils"] = _tree_pipeline
sys.modules[__name__ + ".node_selector"] = _tree_pipeline

__all__: list[str] = []
