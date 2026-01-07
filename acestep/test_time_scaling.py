"""
Test-Time Scaling Module
Implements perplexity-based scoring for generated audio codes
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from loguru import logger
import yaml


def perplexity_to_score(perplexity: float, scale: float = 100.0) -> float:
    """
    Convert perplexity to a normalized score in [0, 1] range.
    
    Lower perplexity = higher score (better quality)
    Uses exponential decay: score = exp(-perplexity / scale)
    
    Args:
        perplexity: Perplexity value (typically 1 to 1000+)
        scale: Scale parameter to control score distribution (default 100.0)
               - Smaller scale: more sensitive to perplexity changes
               - Larger scale: less sensitive to perplexity changes
        
    Returns:
        Score in [0, 1] range, where 1 is perfect and 0 is worst
        
    Examples:
        perplexity=1   → score≈0.99  (excellent)
        perplexity=50  → score≈0.61  (good if scale=100)
        perplexity=100 → score≈0.37  (medium if scale=100)
        perplexity=500 → score≈0.01  (poor if scale=100)
    """
    import math
    return math.exp(-perplexity / scale)


def calculate_perplexity(
    llm_handler,
    audio_codes: str,
    caption: str = "",
    lyrics: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
) -> Tuple[float, str]:
    """
    Calculate perplexity of generated audio codes conditioned on caption/lyrics/metadata.
    
    This reverses the generation task: given audio codes as input, measure how well 
    the model can predict the CoT metadata and lyrics that should generate those codes.
    
    Lower perplexity = model is less surprised = better quality generation
    Score = -perplexity (higher is better)
    
    The understanding task format is:
    Input: <|audio_code_123|><|audio_code_456|>...
    Output: <think>\nmetadata_yaml\n</think>\n\n# Lyric\nlyrics_text
    
    Args:
        llm_handler: LLM handler instance with initialized model
        audio_codes: Generated audio code string (e.g., "<|audio_code_123|><|audio_code_456|>...")
        caption: Caption text used for generation
        lyrics: Lyrics text used for generation
        metadata: Dictionary with CoT metadata fields (bpm, duration, keyscale, language, timesignature, etc.)
        temperature: Temperature for probability scaling (default 1.0)
        
    Returns:
        Tuple of (perplexity_value, status_message)
        
    Example:
        metadata = {'bpm': 120, 'duration': 30, 'keyscale': 'C major', 'language': 'en', 'timesignature': '4'}
        perplexity, status = calculate_perplexity(
            llm_handler, 
            audio_codes="<|audio_code_123|>...",
            caption="calm piano",
            lyrics="verse 1...",
            metadata=metadata
        )
        score = -perplexity  # Higher score = better quality
    """
    if not llm_handler.llm_initialized:
        return float('inf'), "❌ LLM not initialized"
    
    if not audio_codes or not audio_codes.strip():
        return float('inf'), "❌ No audio codes provided"
    
    try:
        # Build the understanding prompt: codes as input
        # The model should generate: <think>metadata</think>\n# Lyric\n...
        formatted_prompt = llm_handler.build_formatted_prompt_for_understanding(
            audio_codes=audio_codes,
            is_negative_prompt=False
        )
        
        logger.info(f"Calculating perplexity for {len(audio_codes)} character audio codes")
        
        # Build the expected output (target sequence) following understanding task format
        # Format: <think>\nmetadata_yaml\n</think>\n\n# Lyric\nlyrics_text
        target_parts = []
        
        # Build CoT section with metadata
        if metadata and isinstance(metadata, dict):
            # Filter out None values and format as YAML (sorted keys)
            cot_items = {}
            for key in ['bpm', 'caption', 'duration', 'genres', 'keyscale', 'language', 'timesignature']:
                if key in metadata and metadata[key] is not None:
                    cot_items[key] = metadata[key]
            
            if cot_items:
                cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()
                target_parts.append(f"<think>\n{cot_yaml}\n</think>\n")
        
        # Add Lyric section (note: understanding task uses "# Lyric" not "# Caption")
        if lyrics:
            target_parts.append(f"\n# Lyric\n{lyrics}\n")
        
        target_text = "".join(target_parts)
        
        if not target_text.strip():
            return float('inf'), "❌ No target text to evaluate (lyrics or metadata required)"
        
        logger.debug(f"Target text (first 200 chars): {target_text[:200]}...")
        
        # Calculate perplexity using appropriate backend
        if llm_handler.llm_backend == "vllm":
            perplexity = _calculate_perplexity_vllm(
                llm_handler,
                formatted_prompt,
                target_text,
                temperature
            )
        else:  # pt backend
            perplexity = _calculate_perplexity_pt(
                llm_handler,
                formatted_prompt,
                target_text,
                temperature
            )
        
        status_msg = f"✅ Perplexity calculated: {perplexity:.4f}"
        logger.info(status_msg)
        return perplexity, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error calculating perplexity: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return float('inf'), error_msg


def _calculate_perplexity_pt(
    llm_handler,
    formatted_prompt: str,
    target_text: str,
    temperature: float
) -> float:
    """
    Calculate perplexity using PyTorch backend.
    
    For vllm backend, this uses a shared-weight HuggingFace model.
    For pt backend, this uses the original model.
    
    Args:
        llm_handler: LLM handler with pt or vllm backend
        formatted_prompt: Formatted input prompt (audio codes)
        target_text: Expected output text (CoT metadata + lyrics)
        temperature: Temperature for probability scaling
        
    Returns:
        Perplexity value
    """
    # Get model for scoring (handles both pt and vllm backends)
    model = llm_handler.get_hf_model_for_scoring()
    tokenizer = llm_handler.llm_tokenizer
    device = llm_handler.device if llm_handler.llm_backend == "pt" else next(model.parameters()).device
    
    # Tokenize prompt and target separately
    prompt_tokens = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    
    target_tokens = tokenizer(
        target_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    
    # Concatenate prompt + target for full sequence
    full_input_ids = torch.cat([
        prompt_tokens['input_ids'],
        target_tokens['input_ids']
    ], dim=1).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(full_input_ids)
    
    # Forward pass to get logits
    with torch.no_grad():
        with llm_handler._load_model_context():
            outputs = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # Get the logits for predicting target tokens
    # Shift logits and labels: logits[i] predicts token[i+1]
    prompt_len = prompt_tokens['input_ids'].shape[1]
    target_len = target_tokens['input_ids'].shape[1]
    
    # Extract logits for positions that predict target tokens
    # logits at positions [prompt_len-1 : prompt_len+target_len-1] predict target tokens
    pred_logits = logits[0, prompt_len-1:prompt_len+target_len-1, :]  # [target_len, vocab_size]
    target_ids = target_tokens['input_ids'][0]  # [target_len]
    
    # Apply temperature scaling
    if temperature != 1.0:
        pred_logits = pred_logits / temperature
    
    # Calculate cross-entropy loss for each position
    log_probs = F.log_softmax(pred_logits, dim=-1)  # [target_len, vocab_size]
    
    # Gather log probabilities of target tokens
    target_log_probs = log_probs[torch.arange(target_len), target_ids]  # [target_len]
    
    # Calculate perplexity: exp(-mean(log_probs))
    mean_neg_log_prob = -target_log_probs.mean()
    perplexity = torch.exp(mean_neg_log_prob).item()
    
    return perplexity


def _calculate_perplexity_vllm(
    llm_handler,
    formatted_prompt: str,
    target_text: str,
    temperature: float
) -> float:
    """
    Calculate perplexity using vllm backend.
    
    Uses shared-weight HuggingFace model for perplexity calculation.
    This avoids the complexity of nanovllm's context management.
    
    Args:
        llm_handler: LLM handler with vllm backend
        formatted_prompt: Formatted input prompt (audio codes)
        target_text: Expected output text (CoT metadata + lyrics)
        temperature: Temperature for probability scaling
        
    Returns:
        Perplexity value
    """
    logger.debug("Using vllm backend with shared-weight HuggingFace model for perplexity")
    # Delegate to pt backend implementation which now handles both backends
    return _calculate_perplexity_pt(llm_handler, formatted_prompt, target_text, temperature)
