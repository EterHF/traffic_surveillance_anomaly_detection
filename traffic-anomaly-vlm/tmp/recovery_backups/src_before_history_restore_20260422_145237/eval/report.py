from __future__ import annotations

from src.core.utils import write_json


def save_report(path: str, report: dict) -> None:
    write_json(path, report)
