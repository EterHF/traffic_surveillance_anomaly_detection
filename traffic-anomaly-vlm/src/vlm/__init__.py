"""Vision-language model modules.

This initializer intentionally avoids importing heavy runtime dependencies
such as `transformers` on package import.
"""

__all__ = [
    "infer",
    "model_loader",
    "parser",
    "prompts",
]
