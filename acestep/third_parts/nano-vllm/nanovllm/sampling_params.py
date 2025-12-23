from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    cfg_scale: float = 1.0  # CFG guidance scale. When > 1.0, applies classifier-free guidance
    top_k: Optional[int] = None  # Top-k sampling: consider only top k tokens
    top_p: Optional[float] = None  # Top-p (nucleus) sampling: consider tokens with cumulative probability <= top_p
    repetition_penalty: float = 1.0  # Repetition penalty: >1.0 reduces repetition, <1.0 increases it

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.cfg_scale >= 1.0, "cfg_scale must be >= 1.0"
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be > 0"
        if self.top_p is not None:
            assert 0.0 < self.top_p <= 1.0, "top_p must be in (0.0, 1.0]"
        assert self.repetition_penalty > 0.0, "repetition_penalty must be > 0.0"
