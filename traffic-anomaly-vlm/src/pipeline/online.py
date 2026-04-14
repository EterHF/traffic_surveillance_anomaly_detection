from __future__ import annotations


class OnlinePipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, source: str):
        raise NotImplementedError("Online pipeline is reserved for streaming integration.")
