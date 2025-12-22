import os
import socket
from dataclasses import dataclass
from transformers import AutoConfig


def find_available_port(start_port: int = 2333, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: The starting port number to check
        max_attempts: Maximum number of ports to try
        
    Returns:
        An available port number
        
    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return port
        except OSError:
            # Port is in use, try next one
            continue
    raise RuntimeError(f"Could not find an available port starting from {start_port} after {max_attempts} attempts")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = 151643
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    dist_port: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        # Auto-find available port if not specified
        if self.dist_port is None:
            self.dist_port = find_available_port()
