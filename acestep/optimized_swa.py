import torch
import torch.nn.functional as F
import math

def optimized_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scaling: float = None,
) -> torch.Tensor:
    """
    Block-wise Sliding Window Attention implementation using PyTorch Eager Mode.
    
    Args:
        query: [Batch, Heads, Seq_Len, Head_Dim]
        key:   [Batch, Heads, Seq_Len, Head_Dim]
        value: [Batch, Heads, Seq_Len, Head_Dim]
        window_size: int, sliding window radius (one-sided)
        scaling: float, scaling factor for attention scores (default: 1 / sqrt(head_dim))
        
    Returns:
        output: [Batch, Heads, Seq_Len, Head_Dim]
    """
    b, h, l, d = query.shape
    
    if scaling is None:
        scaling = 1.0 / math.sqrt(d)
        
    # 1. Padding Query to be multiple of window_size
    pad_len = (window_size - (l % window_size)) % window_size
    if pad_len > 0:
        query = F.pad(query, (0, 0, 0, pad_len))
        # We also need to pad key/value to match length for the main structure, 
        # though we will add extra padding for the window later.
        # Actually, for K/V, we just need them to be long enough to cover the windows.
        # Let's pad them to match Q's padded length first to simplify indexing.
        key = F.pad(key, (0, 0, 0, pad_len))
        value = F.pad(value, (0, 0, 0, pad_len))
    
    l_padded = query.shape[2]
    num_chunks = l_padded // window_size
    
    # 2. Prepare Key/Value with halo padding
    # We need [i*W - W : (i+1)*W + W] for each chunk i.
    # So we pad W on both sides of the sequence dimension.
    # K shape: [B, H, L_padded, D] -> [B, H, W + L_padded + W, D]
    key_padded = F.pad(key, (0, 0, window_size, window_size))
    value_padded = F.pad(value, (0, 0, window_size, window_size))
    
    # 3. Chunking Query
    # [B, H, L_padded, D] -> [B, H, Num_Chunks, W, D]
    query_chunks = query.view(b, h, num_chunks, window_size, d)
    
    # 4. Unfolding Key/Value
    # We want windows of size 3*W with stride W.
    # Input dim: [B, H, L_padded + 2W, D]
    # Unfold on dim 2.
    # Result: [B, H, Num_Chunks, D, 3*W]
    key_chunks = key_padded.unfold(2, 3 * window_size, window_size)
    value_chunks = value_padded.unfold(2, 3 * window_size, window_size)
    
    # Adjust shapes for matmul: [B, H, Num_Chunks, 3*W, D]
    key_chunks = key_chunks.transpose(-1, -2)
    value_chunks = value_chunks.transpose(-1, -2)
    
    # 5. Attention Scores
    # Q: [..., W, D], K: [..., 3W, D] -> Scores: [..., W, 3W]
    scores = torch.matmul(query_chunks, key_chunks.transpose(-1, -2)) * scaling
    
    # 6. Apply Local Mask
    # Construct mask once
    # q_idx in [0, W), k_idx in [0, 3W)
    # Valid if k_idx in [q_idx, q_idx + 2W]
    
    local_q_idx = torch.arange(window_size, device=query.device).unsqueeze(1) # [W, 1]
    local_k_idx = torch.arange(3 * window_size, device=query.device).unsqueeze(0) # [1, 3W]
    
    # Geometric mask
    mask = (local_k_idx >= local_q_idx) & (local_k_idx <= (local_q_idx + 2 * window_size))
    # [1, 1, 1, W, 3W]
    mask = mask.view(1, 1, 1, window_size, 3 * window_size)
    
    # Padding mask
    # We need to mask out keys that are padding (either halo or alignment padding)
    # Valid keys in key_padded are at indices [window_size, window_size + l)
    valid_key_mask = torch.zeros(l_padded + 2 * window_size, device=query.device, dtype=torch.bool)
    valid_key_mask[window_size : window_size + l] = True
    
    # Unfold to match key_chunks: [Num_Chunks, 3W]
    valid_key_mask_chunks = valid_key_mask.unfold(0, 3 * window_size, window_size)
    # Reshape to broadcast: [1, 1, Num_Chunks, 1, 3W]
    valid_key_mask_chunks = valid_key_mask_chunks.view(1, 1, num_chunks, 1, 3 * window_size)
    
    # Combine masks
    mask = mask & valid_key_mask_chunks
    
    # Apply mask
    min_dtype = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, min_dtype)
    
    # 7. Softmax and Weighted Sum
    attn_probs = F.softmax(scores, dim=-1)
    # [..., W, 3W] @ [..., 3W, D] -> [..., W, D]
    output_chunks = torch.matmul(attn_probs, value_chunks)
    
    # 8. Reshape and Crop
    # [B, H, Num_Chunks, W, D] -> [B, H, L_padded, D]
    output = output_chunks.view(b, h, l_padded, d)
    
    # Remove padding
    if pad_len > 0:
        output = output[:, :, :l, :]
        
    return output
