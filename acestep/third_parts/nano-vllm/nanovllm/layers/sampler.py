import torch
from torch import nn
from typing import Optional


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self, 
        logits: torch.Tensor, 
        temperatures: torch.Tensor,
        top_ks: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        repetition_penalties: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Sample tokens from logits with optional top-k, top-p, and repetition penalty.
        
        Args:
            logits: [batch_size, vocab_size] logits tensor
            temperatures: [batch_size] temperature values
            top_ks: Optional [batch_size] top-k values (None or 0 means no top-k filtering)
            top_ps: Optional [batch_size] top-p values (None or 1.0 means no top-p filtering)
            repetition_penalties: Optional [batch_size] repetition penalty values (1.0 means no penalty)
            input_ids: Optional [batch_size, seq_len] input token ids for repetition penalty
        """
        batch_size, vocab_size = logits.shape
        
        # Note: Repetition penalty is applied in ModelRunner before calling sampler
        # This allows us to use the full sequence context
        
        # Apply temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # Apply top-k filtering if specified
        if top_ks is not None:
            for i in range(batch_size):
                top_k = top_ks[i].item()
                if top_k > 0 and top_k < vocab_size:
                    # Get top-k logits, set others to -inf
                    top_k_logits, top_k_indices = torch.topk(logits[i], int(top_k), dim=-1)
                    filtered_logits = torch.full_like(logits[i], float('-inf'))
                    filtered_logits[top_k_indices] = top_k_logits
                    logits[i] = filtered_logits
        
        # Apply top-p (nucleus) filtering if specified
        if top_ps is not None:
            probs = torch.softmax(logits, dim=-1)
            for i in range(batch_size):
                top_p = top_ps[i].item()
                if 0.0 < top_p < 1.0:
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                    # Calculate cumulative probabilities
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Find the cutoff point
                    cutoff_idx = (cumsum_probs <= top_p).sum().item()
                    if cutoff_idx < len(sorted_indices):
                        cutoff_idx += 1  # Include one more token to ensure we have at least one
                    # Create mask for tokens to keep
                    mask = torch.zeros_like(probs[i])
                    mask[sorted_indices[:cutoff_idx]] = 1.0
                    # Apply mask: set filtered tokens to -inf
                    logits[i] = torch.where(mask > 0, logits[i], torch.tensor(float('-inf'), device=logits.device))
        
        # Sample using Gumbel-max trick (equivalent to sampling from softmax)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
