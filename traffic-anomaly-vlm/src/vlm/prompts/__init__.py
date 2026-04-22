"""Prompt builders for VLM-based traffic understanding.

Read these files in this order:
1. `stage1.py`: neutral scene description prompts
2. `stage2.py`: anomaly judgement prompts
3. `span_score.py`: reusable span scoring prompt for coarse/fine candidates
"""

from .span_score import build_span_score_prompt
from .stage1 import build_pure_stage1_prompt, build_stage1_prompt
from .stage2 import build_pure_stage2_prompt, build_stage2_prompt

__all__ = [
    "build_pure_stage1_prompt",
    "build_pure_stage2_prompt",
    "build_span_score_prompt",
    "build_stage1_prompt",
    "build_stage2_prompt",
]
