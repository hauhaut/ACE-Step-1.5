from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    cfg_scale: float = 1.0  # CFG guidance scale. When > 1.0, applies classifier-free guidance

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.cfg_scale >= 1.0, "cfg_scale must be >= 1.0"
