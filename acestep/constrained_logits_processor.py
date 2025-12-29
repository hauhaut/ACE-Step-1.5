
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Callable, Set
from loguru import logger
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
import os
import torch


# ==============================================================================
# FSM States for Constrained Decoding
# ==============================================================================
class FSMState(Enum):
    """Finite State Machine states for metadata generation"""
    THINK_TAG = auto()           # Generating "<think>"
    NEWLINE_AFTER_THINK = auto() # Generating "\n" after <think>
    BPM_NAME = auto()            # Generating "bpm: "
    BPM_VALUE = auto()           # Generating numeric value 30-300
    NEWLINE_AFTER_BPM = auto()   # Generating "\n" after bpm value
    DURATION_NAME = auto()       # Generating "duration: "
    DURATION_VALUE = auto()      # Generating numeric value 10-600
    NEWLINE_AFTER_DURATION = auto()
    GENRES_NAME = auto()         # Generating "genres: "
    GENRES_VALUE = auto()        # Generating any non-empty string
    NEWLINE_AFTER_GENRES = auto()
    KEYSCALE_NAME = auto()       # Generating "keyscale: "
    KEYSCALE_VALUE = auto()      # Generating keyscale pattern
    NEWLINE_AFTER_KEYSCALE = auto()
    TIMESIG_NAME = auto()        # Generating "timesignature: "
    TIMESIG_VALUE = auto()       # Generating 2, 3, 4, or 6
    NEWLINE_AFTER_TIMESIG = auto()
    THINK_END_TAG = auto()       # Generating "</think>"
    CODES_GENERATION = auto()    # Generating audio codes (no constraints)
    COMPLETED = auto()           # Generation completed


class MetadataConstrainedLogitsProcessor(LogitsProcessor):
    """
    FSM-driven LogitsProcessor that constrains generation to produce valid metadata.
    
    This processor enforces the following format:
    <think>
    bpm: [30-300]
    duration: [10-600]
    genres: [any non-empty string]
    keyscale: [A-G][#/♭]? [major/minor]
    timesignature: [2/3/4/6]
    </think>
    
    It uses token masking (setting invalid token logits to -inf) to enforce constraints.
    For numeric fields, it uses early-blocking to prevent out-of-range values.
    For field transitions (e.g., end of numeric value), it compares P(newline) vs P(digit).
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        enabled: bool = True,
        debug: bool = False,
        genres_vocab_path: Optional[str] = None,
        skip_genres: bool = True,
    ):
        """
        Initialize the constrained logits processor.
        
        This processor should be initialized once when loading the LLM and reused
        for all generations. Use update_caption() before each generation to update
        the caption-based genre filtering.
        
        Args:
            tokenizer: The tokenizer to use for encoding/decoding
            enabled: Whether to enable constrained decoding
            debug: Whether to print debug information
            genres_vocab_path: Path to genres vocabulary file (one genre per line)
                              If None, defaults to "acestep/genres_vocab.txt"
            skip_genres: Whether to skip genres generation in metadata (default True)
        """
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.debug = debug
        self.skip_genres = skip_genres
        self.caption: Optional[str] = None  # Set via update_caption() before each generation
        
        # User-provided metadata fields (optional)
        # If provided, these fields will be used directly instead of generating
        # Format: {"bpm": "120", "duration": "234", "keyscale": "G major", "timesignature": "4", "genres": "Pop Rock"}
        self.user_provided_metadata: Dict[str, Optional[str]] = {
            "bpm": None,
            "duration": None,
            "keyscale": None,
            "timesignature": None,
            "genres": None,
        }
        
        # Temperature settings for different generation phases (set per-generation)
        # If set, the processor will apply temperature scaling (divide logits by temperature)
        # Note: Set base sampler temperature to 1.0 when using processor-based temperature
        self.metadata_temperature: Optional[float] = None
        self.codes_temperature: Optional[float] = None
        
        # Duration constraint for codes generation
        # 5 codes = 1 second, so target_codes = target_duration * 5
        self.target_duration: Optional[float] = None  # User-specified duration in seconds
        self.target_codes: Optional[int] = None  # Computed target codes count
        self.codes_count: int = 0  # Counter for generated codes
        
        # Stop at reasoning flag - if True, stop generation after </think> tag
        self.stop_at_reasoning: bool = False
        
        # Current state
        self.state = FSMState.THINK_TAG
        self.position_in_state = 0  # Position within current state's fixed string
        self.accumulated_value = ""  # For numeric/text value accumulation (legacy, for compatibility)
        self.accumulated_token_ids: List[int] = []  # Token ID sequence for keyscale (and other fields)
        
        # Token queue for user-provided fields (injected directly without generation)
        self.user_field_token_queue: List[int] = []
        self.current_user_field: Optional[str] = None  # Current field being injected
        
        # Pre-compute token IDs for efficiency
        self._precompute_tokens()
        
        # Genres vocabulary for constrained decoding
        self.genres_vocab_path = genres_vocab_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "genres_vocab.txt"
        )
        self.genres_vocab: List[str] = []  # Full vocab
        self.genres_vocab_mtime: float = 0.0
        self.genres_trie: Dict = {}  # Trie for full vocab (fallback)
        self.caption_genres_trie: Dict = {}  # Trie for caption-matched genres (priority)
        self.caption_matched_genres: List[str] = []  # Genres matched from caption
        self._char_to_tokens: Dict[str, set] = {}  # Precomputed char -> token IDs mapping
        
        # Precompute token mappings once (O(vocab_size), runs once at init)
        self._precompute_char_token_mapping()
        
        # Field definitions (needed before building prefix trees)
        self.field_specs = {
            "bpm": {"min": 30, "max": 300},
            "duration": {"min": 10, "max": 600},
            "timesignature": {"valid_values": [2, 3, 4, 6]},
        }
        
        # Build valid numeric values for BPM, Duration, Timesignature
        # These will be used to build prefix trees based on actual tokenization
        self.valid_bpm_values = [str(v) for v in range(self.field_specs["bpm"]["min"], self.field_specs["bpm"]["max"] + 1)]
        self.valid_duration_values = [str(v) for v in range(self.field_specs["duration"]["min"], self.field_specs["duration"]["max"] + 1)]
        self.valid_timesig_values = [str(v) for v in self.field_specs["timesignature"]["valid_values"]]
        
        # Build keyscale prefix tree (requires _char_to_tokens to be initialized)
        self.keyscale_prefix_tree = self._build_keyscale_prefix_tree()
        
        # Build numeric prefix trees (BPM, Duration, Timesignature) with context
        # IMPORTANT: State machine generates "bpm:" (no space), but tokenizer sees "bpm: " (with space)
        # Use same logic as keyscale: context_prefix_for_matching (no space) and context_prefix_for_tokenization (with space)
        self.bpm_prefix_tree = self._build_numeric_prefix_tree(
            self.valid_bpm_values, 
            context_prefix_for_matching="bpm:",
            context_prefix_for_tokenization="bpm: "
        )
        self.duration_prefix_tree = self._build_numeric_prefix_tree(
            self.valid_duration_values,
            context_prefix_for_matching="duration:",
            context_prefix_for_tokenization="duration: "
        )
        self.timesig_prefix_tree = self._build_numeric_prefix_tree(
            self.valid_timesig_values,
            context_prefix_for_matching="timesignature:",
            context_prefix_for_tokenization="timesignature: "
        )
        
        self._load_genres_vocab()
        
        # Note: Caption-based genre filtering is initialized via update_caption() before each generation
        
        # Fixed strings for each state
        # IMPORTANT: Do NOT include trailing space after colon - tokenizer will handle spacing
        # All matching should be done at token level, not string level
        # NOTE: NEWLINE_AFTER_* states are removed - field values generate newline directly and transition to next field
        self.fixed_strings = {
            FSMState.THINK_TAG: "<think>",
            FSMState.NEWLINE_AFTER_THINK: "\n",
            FSMState.BPM_NAME: "bpm:",
            FSMState.DURATION_NAME: "duration:",
            FSMState.GENRES_NAME: "genres:",
            FSMState.KEYSCALE_NAME: "keyscale:",
            FSMState.TIMESIG_NAME: "timesignature:",
            FSMState.THINK_END_TAG: "</think>",
        }
        
        # State transitions - build dynamically based on skip_genres
        self._build_state_transitions()
    
    def _get_next_field_state(self, current_field: str) -> Optional[FSMState]:
        """
        Get the next field state. Always returns the next field's NAME state,
        even if the field is user-provided (we still need to generate the field name).
        
        Args:
            current_field: Current field name ("bpm", "duration", "genres", "keyscale", "timesignature")
            
        Returns:
            Next FSMState (NAME state of next field), or THINK_END_TAG if no more fields
        """
        field_order = ["bpm", "duration", "genres", "keyscale", "timesignature"]
        field_to_state = {
            "bpm": FSMState.BPM_NAME,
            "duration": FSMState.DURATION_NAME,
            "genres": FSMState.GENRES_NAME,
            "keyscale": FSMState.KEYSCALE_NAME,
            "timesignature": FSMState.TIMESIG_NAME,
        }
        
        try:
            current_idx = field_order.index(current_field)
        except ValueError:
            return FSMState.THINK_END_TAG
        
        # Find next field in order
        for i in range(current_idx + 1, len(field_order)):
            field = field_order[i]
            
            # Skip genres if skip_genres is True
            if field == "genres" and self.skip_genres:
                continue
            
            # Return the next field's NAME state (even if user-provided, we still generate field name)
            return field_to_state[field]
        
        # No more fields, go to THINK_END_TAG
        return FSMState.THINK_END_TAG
    
    def _build_state_transitions(self):
        """Build state transition map based on skip_genres and user-provided metadata."""
        self.next_state = {
            FSMState.THINK_TAG: FSMState.NEWLINE_AFTER_THINK,
            FSMState.NEWLINE_AFTER_THINK: FSMState.BPM_NAME,  # Always start with BPM
            FSMState.THINK_END_TAG: FSMState.CODES_GENERATION,
            FSMState.CODES_GENERATION: FSMState.COMPLETED,
        }
        
        # Build transitions for all fields (even if user-provided, we still need to generate field name)
        # Field order: bpm -> duration -> genres -> keyscale -> timesignature
        
        # BPM field: NAME -> VALUE -> next field
        self.next_state[FSMState.BPM_NAME] = FSMState.BPM_VALUE
        self.next_state[FSMState.BPM_VALUE] = self._get_next_field_state("bpm")
        
        # Duration field: NAME -> VALUE -> next field
        self.next_state[FSMState.DURATION_NAME] = FSMState.DURATION_VALUE
        self.next_state[FSMState.DURATION_VALUE] = self._get_next_field_state("duration")
        
        # Genres field (only if not skipped): NAME -> VALUE -> next field
        if not self.skip_genres:
            self.next_state[FSMState.GENRES_NAME] = FSMState.GENRES_VALUE
            self.next_state[FSMState.GENRES_VALUE] = self._get_next_field_state("genres")
        
        # Keyscale field: NAME -> VALUE -> next field
        self.next_state[FSMState.KEYSCALE_NAME] = FSMState.KEYSCALE_VALUE
        self.next_state[FSMState.KEYSCALE_VALUE] = self._get_next_field_state("keyscale")
        
        # Timesignature field: NAME -> VALUE -> THINK_END_TAG
        self.next_state[FSMState.TIMESIG_NAME] = FSMState.TIMESIG_VALUE
        self.next_state[FSMState.TIMESIG_VALUE] = FSMState.THINK_END_TAG
    
    def set_skip_genres(self, skip: bool):
        """Set whether to skip genres generation and rebuild state transitions."""
        self.skip_genres = skip
        self._build_state_transitions()
    
    def set_stop_at_reasoning(self, stop: bool):
        """
        Set whether to stop generation after </think> tag.
        
        Args:
            stop: If True, generation will stop immediately after </think> tag is generated.
                  If False, generation continues to codes generation phase.
        """
        self.stop_at_reasoning = stop
    
    def set_user_metadata(self, metadata: Optional[Dict[str, Optional[str]]] = None):
        """
        Set user-provided metadata fields. Fields that are provided will be used directly
        instead of generating. Fields that are None will be generated.
        
        Args:
            metadata: Dictionary with optional fields:
                - "bpm": Optional[str] - e.g., "120"
                - "duration": Optional[str] - e.g., "234"
                - "keyscale": Optional[str] - e.g., "G major"
                - "timesignature": Optional[str] - e.g., "4"
                - "genres": Optional[str] - e.g., "Pop Rock"
                If None, clears all user-provided metadata.
        """
        if metadata is None:
            metadata = {}
        
        # Update user-provided metadata
        for field in ["bpm", "duration", "keyscale", "timesignature", "genres"]:
            if field in metadata:
                self.user_provided_metadata[field] = metadata[field]
            else:
                self.user_provided_metadata[field] = None
        
        # Rebuild state transitions to skip provided fields
        self._build_state_transitions()
        
        if self.debug:
            provided_fields = [k for k, v in self.user_provided_metadata.items() if v is not None]
            if provided_fields:
                logger.debug(f"User provided metadata fields: {provided_fields}")
            else:
                logger.debug("No user-provided metadata, all fields will be generated")
    
    def _precompute_tokens(self):
        """Pre-compute commonly used token IDs for efficiency."""
        # Digit tokens (0-9)
        self.digit_tokens = {}
        for d in range(10):
            tokens = self.tokenizer.encode(str(d), add_special_tokens=False)
            if tokens:
                self.digit_tokens[d] = tokens[-1]  # Take last token (in case of prefix)
        
        # Newline token
        newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.newline_token = newline_tokens[-1] if newline_tokens else None
        
        # Note tokens for keyscale (A-G)
        self.note_tokens = {}
        for note in "ABCDEFG":
            tokens = self.tokenizer.encode(note, add_special_tokens=False)
            if tokens:
                self.note_tokens[note] = tokens[-1]
        
        # Sharp/flat tokens
        self.sharp_tokens = []
        for s in ["#", "♯"]:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            if tokens:
                self.sharp_tokens.append(tokens[-1])
        
        self.flat_tokens = []
        for f in ["b", "♭"]:
            tokens = self.tokenizer.encode(f, add_special_tokens=False)
            if tokens:
                self.flat_tokens.append(tokens[-1])
        
        # Space token
        space_tokens = self.tokenizer.encode(" ", add_special_tokens=False)
        self.space_token = space_tokens[-1] if space_tokens else None
        
        # Major/minor tokens (we'll encode the full words)
        self.major_start_tokens = []
        self.minor_start_tokens = []
        for prefix in ["m", "M"]:
            tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            if tokens:
                if prefix.lower() == "m":
                    self.minor_start_tokens.append(tokens[-1])
                    self.major_start_tokens.append(tokens[-1])  # "major" also starts with m
        
        # Vocab size
        self.vocab_size = len(self.tokenizer)
        
        # Comma token for multi-genre support
        comma_tokens = self.tokenizer.encode(",", add_special_tokens=False)
        self.comma_token = comma_tokens[-1] if comma_tokens else None
        
        # EOS token for duration-constrained codes generation
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Build valid keyscales set (prefix tree will be built after _char_to_tokens is initialized)
        # 7 notes × 5 accidentals (none, #, b, ♯, ♭) × 2 modes = 70 valid combinations
        notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        accidentals = ['', '#', 'b', '♯', '♭']  # empty + ASCII sharp/flat + Unicode sharp/flat
        modes = ['major', 'minor']
        
        self.valid_keyscales = set()
        for note in notes:
            for acc in accidentals:
                for mode in modes:
                    self.valid_keyscales.add(f"{note}{acc} {mode}")
        
        # keyscale_prefix_tree will be built in _precompute_char_token_mapping() after _char_to_tokens is ready
        # Numeric prefix trees will be built after field_specs is defined
    
    def _build_keyscale_prefix_tree(self) -> Dict[Tuple[int, ...], Set[int]]:
        """
        Build keyscale prefix to allowed tokens mapping based on ACTUAL tokenization.
        
        IMPORTANT: Uses token ID sequences as keys, NOT strings, to avoid tokenization mismatches.
        
        CRITICAL FIX: The tokenizer may merge the context's trailing space into the next token.
        For example:
        - "keyscale: " tokenizes to [10563, 2246, 25, 220] -> ['keys', 'cale', ':', ' ']
        - "keyscale: G major" tokenizes to [10563, 2246, 25, 479, 3598] -> ['keys', 'cale', ':', ' G', ' major']
        The space ' ' (220) is merged into ' G' (479), so we can't use simple slicing.
        
        Strategy:
        1. For each keyscale (e.g., "G major"), encode the FULL string "keyscale: G major"
        2. Tokenize to get: [10563, 2246, 25, 479, 3598] -> ['keys', 'cale', ':', ' G', ' major']
        3. Find where context prefix ends by matching token sequences (handling space merging)
        4. Extract keyscale value tokens: [479, 3598] (for "G major")
        5. Build prefix tree using token ID sequences as keys
        
        This ensures we get the exact tokenization that occurs during generation.
        """
        prefix_to_tokens: Dict[Tuple[int, ...], Set[int]] = {}
        
        # Context prefix that appears before keyscale value
        # IMPORTANT: The state machine generates "keyscale:" (no space), but when tokenizing
        # the full string "keyscale: G major", the tokenizer includes space, so we need to
        # match the actual tokenization behavior.
        # 
        # Strategy:
        # 1. Use "keyscale:" (no space) to match the state machine's output
        # 2. But when building prefix tree, use "keyscale: " (with space) + keyscale to match actual tokenization
        context_prefix_for_matching = "keyscale:"  # What state machine generates
        context_prefix_for_tokenization = "keyscale: "  # What tokenizer sees in full string
        
        # First, tokenize the context (without space) to know its token sequence for matching
        context_token_ids = self.tokenizer.encode(context_prefix_for_matching, add_special_tokens=False)
        
        if self.debug:
            context_tokens_str = [self.tokenizer.decode([t]) for t in context_token_ids]
            logger.debug(f"Context for matching 'keyscale:' tokenizes to {context_token_ids} -> {context_tokens_str}")
        
        # For each valid keyscale, encode full string and extract value tokens
        for keyscale in self.valid_keyscales:
            # Step 1: Encode full string "keyscale: {keyscale}" (with space, as tokenizer sees it)
            full_text = context_prefix_for_tokenization + keyscale
            full_token_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # Step 2: Find where context ends in full_token_ids
            # We match using context_prefix_for_matching ("keyscale:") token sequence
            # because that's what the state machine actually generates
            context_end_idx = None
            
            # Try exact prefix match using context_prefix_for_matching token sequence
            if len(full_token_ids) >= len(context_token_ids):
                if full_token_ids[:len(context_token_ids)] == context_token_ids:
                    context_end_idx = len(context_token_ids)
            
            if context_end_idx is None:
                if self.debug:
                    logger.warning(f"Could not find context prefix in full tokenization of '{full_text}', skipping")
                continue
            
            # Step 3: Extract keyscale value tokens (everything after context)
            keyscale_token_ids = full_token_ids[context_end_idx:]
            
            # Step 4: Verify we extracted some tokens (sanity check)
            if not keyscale_token_ids:
                if self.debug:
                    logger.warning(f"No tokens extracted for keyscale '{keyscale}', skipping")
                continue
            
            # Step 5: Verify first token is a note (A-G)
            # This is critical: the first token of keyscale value must be a note
            first_token_id = keyscale_token_ids[0]
            first_token_str = self.tokenizer.decode([first_token_id])
            # Check if first token starts with a note (A-G, case insensitive, with optional leading space)
            first_char = first_token_str.lstrip()[0].upper() if first_token_str.lstrip() else ""
            if first_char not in "ABCDEFG":
                # This keyscale's first token is not a note - skip it
                if self.debug:
                    logger.debug(f"Skipping keyscale '{keyscale}': first token is '{first_token_str}' (id={first_token_id}), not a note")
                continue
            
            # Step 6: Build prefix mappings from keyscale value tokens
            # Use token ID sequences as keys (not strings) to avoid tokenization mismatches
            for i in range(len(keyscale_token_ids) + 1):
                # Current token sequence prefix (empty tuple for start)
                token_prefix = tuple(keyscale_token_ids[:i])
                
                if token_prefix not in prefix_to_tokens:
                    prefix_to_tokens[token_prefix] = set()
                
                if i < len(keyscale_token_ids):
                    # Add next token as allowed for current prefix
                    next_token_id = keyscale_token_ids[i]
                    prefix_to_tokens[token_prefix].add(next_token_id)
                else:
                    # Complete keyscale should allow newline
                    if self.newline_token:
                        prefix_to_tokens[token_prefix].add(self.newline_token)
        
        if self.debug:
            logger.debug(f"Built keyscale prefix tree with {len(prefix_to_tokens)} token sequence prefixes")
            # Check empty prefix (start of keyscale value)
            empty_prefix = tuple()
            if empty_prefix in prefix_to_tokens:
                first_tokens = prefix_to_tokens[empty_prefix]
                decoded_first = [(t, repr(self.tokenizer.decode([t]))) for t in sorted(first_tokens)]
                logger.debug(f"First tokens allowed (empty prefix): {decoded_first}")
        
        return prefix_to_tokens
    
    def _build_numeric_prefix_tree(
        self, 
        valid_values: List[str], 
        context_prefix_for_matching: str = "",
        context_prefix_for_tokenization: str = ""
    ) -> Dict[Tuple[int, ...], Set[int]]:
        """
        Build prefix tree for numeric field based on actual tokenization with context.
        
        IMPORTANT: Uses token ID sequences as keys, NOT strings, to avoid tokenization mismatches.
        
        Args:
            valid_values: List of valid numeric strings (e.g., ["30", "31", ..., "300"])
            context_prefix_for_matching: Context string that state machine generates (e.g., "bpm:") - no space
            context_prefix_for_tokenization: Context string for tokenization (e.g., "bpm: ") - with space
            
        Returns:
            Dict mapping token ID sequence prefix -> set of allowed token IDs
        """
        prefix_to_tokens: Dict[Tuple[int, ...], Set[int]] = {}
        
        # Encode context for matching (what state machine generates, no space)
        context_token_ids = self.tokenizer.encode(context_prefix_for_matching, add_special_tokens=False) if context_prefix_for_matching else []
        
        # For each valid value, encode it with context and build prefix mappings
        for value_str in valid_values:
            # Encode value WITH context (with space) to match actual tokenization
            full_text = context_prefix_for_tokenization + value_str
            token_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # Find where context ends in full_token_ids using context_prefix_for_matching token sequence
            context_end_idx = None
            if len(token_ids) >= len(context_token_ids):
                if token_ids[:len(context_token_ids)] == context_token_ids:
                    context_end_idx = len(context_token_ids)
            
            if context_end_idx is None:
                if self.debug:
                    logger.warning(f"Could not find context prefix in full tokenization of '{full_text}', skipping")
                continue
            
            # Extract only tokens that belong to the value itself (skip context tokens)
            value_token_ids = token_ids[context_end_idx:]
            
            # Build prefix mappings using token ID sequences as keys
            for i in range(len(value_token_ids) + 1):
                # Current token sequence prefix (empty tuple for start)
                token_prefix = tuple(value_token_ids[:i])
                
                if token_prefix not in prefix_to_tokens:
                    prefix_to_tokens[token_prefix] = set()
                
                if i < len(value_token_ids):
                    # Add next token as allowed for current prefix
                    next_token_id = value_token_ids[i]
                    prefix_to_tokens[token_prefix].add(next_token_id)
                else:
                    # Complete value should allow newline
                    if self.newline_token:
                        prefix_to_tokens[token_prefix].add(self.newline_token)
        
        return prefix_to_tokens
    
    def diagnose_keyscale_prefix_tree(self):
        """
        Diagnose the keyscale prefix tree to help debug generation bias.
        Call this method to print detailed information about allowed tokens at each prefix.
        """
        print("=" * 60)
        print("KEYSCALE PREFIX TREE DIAGNOSIS")
        print("=" * 60)
        
        # Check empty prefix (first token)
        if "" in self.keyscale_prefix_tree:
            first_tokens = self.keyscale_prefix_tree[""]
            print(f"\n[Empty prefix] Allowed first tokens ({len(first_tokens)} total):")
            for t in sorted(first_tokens):
                decoded = self.tokenizer.decode([t])
                print(f"  Token {t}: {repr(decoded)}")
        else:
            print("\nWARNING: Empty prefix not in tree!")
        
        # Check some common prefixes
        test_prefixes = ["A", "B", "C", "D", "E", "F", "G"]
        for prefix in test_prefixes:
            # Try both with and without potential tokenizer artifacts
            for test_key in [prefix, prefix + " "]:
                if test_key in self.keyscale_prefix_tree:
                    tokens = self.keyscale_prefix_tree[test_key]
                    print(f"\n[Prefix {repr(test_key)}] Allowed tokens ({len(tokens)}):")
                    for t in sorted(tokens):
                        decoded = self.tokenizer.decode([t])
                        print(f"  Token {t}: {repr(decoded)}")
        
        # Show some complete keyscales that should be valid
        print(f"\n[Valid keyscales] Total: {len(self.valid_keyscales)}")
        sample = sorted(list(self.valid_keyscales))[:10]
        for ks in sample:
            print(f"  {repr(ks)}")
        
        print("=" * 60)
    
    def _load_genres_vocab(self):
        """
        Load genres vocabulary from file. Supports hot reload by checking file mtime.
        File format: one genre per line, lines starting with # are comments.
        """
        if not os.path.exists(self.genres_vocab_path):
            if self.debug:
                logger.debug(f"Genres vocab file not found: {self.genres_vocab_path}")
            return
        
        try:
            mtime = os.path.getmtime(self.genres_vocab_path)
            if mtime <= self.genres_vocab_mtime:
                return  # File hasn't changed
            
            with open(self.genres_vocab_path, 'r', encoding='utf-8') as f:
                genres = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        genres.append(line.lower())
                
                self.genres_vocab = genres
                self.genres_vocab_mtime = mtime
                self._build_genres_trie()
                
                if self.debug:
                    logger.debug(f"Loaded {len(self.genres_vocab)} genres from {self.genres_vocab_path}")
        except Exception as e:
            logger.warning(f"Failed to load genres vocab: {e}")
    
    def _build_genres_trie(self):
        """
        Build a trie (prefix tree) from genres vocabulary for efficient prefix matching.
        Each node is a dict with:
          - '_end': True if this node represents a complete genre
          - other keys: next characters in the trie
        """
        self.genres_trie = {}
        
        for genre in self.genres_vocab:
            node = self.genres_trie
            for char in genre:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['_end'] = True  # Mark end of a complete genre
        
        if self.debug:
            logger.debug(f"Built genres trie with {len(self.genres_vocab)} entries")
    
    def _extract_caption_genres(self, caption: str):
        """
        Extract genres from the user's caption that match entries in the vocabulary.
        This creates a smaller trie for faster and more relevant genre generation.
        
        Strategy (optimized - O(words * max_genre_len) instead of O(vocab_size)):
        1. Extract words/phrases from caption
        2. For each word, use trie to find all vocab entries that START with this word
        3. Build a separate trie from matched genres
        """
        if not caption or not self.genres_vocab:
            return
        
        caption_lower = caption.lower()
        matched_genres = set()
        
        # Extract words from caption (split by common delimiters)
        import re
        words = re.split(r'[,\s\-_/\\|]+', caption_lower)
        words = [w.strip() for w in words if w.strip() and len(w.strip()) >= 2]
        
        # For each word, find genres in trie that start with this word
        for word in words:
            # Find all genres starting with this word using trie traversal
            node = self._get_genres_trie_node(word)
            if node is not None:
                # Collect all complete genres under this node
                self._collect_complete_genres(node, word, matched_genres)
        
        # Also check if any word appears as a substring in short genres (< 20 chars)
        # This is a quick check for common single-word genres
        genres_set = set(self.genres_vocab)
        for word in words:
            if word in genres_set:
                matched_genres.add(word)
        
        if not matched_genres:
            if self.debug:
                logger.debug(f"No genres matched in caption, using full vocab")
            return
        
        # Build a trie from matched genres
        self.caption_matched_genres = list(matched_genres)
        self.caption_genres_trie = {}
        
        for genre in matched_genres:
            node = self.caption_genres_trie
            for char in genre:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['_end'] = True
        
        if self.debug:
            logger.debug(f"Matched {len(matched_genres)} genres from caption: {list(matched_genres)[:5]}...")
    
    def _collect_complete_genres(self, node: Dict, prefix: str, result: set, max_depth: int = 50):
        """
        Recursively collect all complete genres under a trie node.
        Limited depth to avoid too many matches.
        """
        if max_depth <= 0:
            return
        
        if node.get('_end', False):
            result.add(prefix)
        
        # Limit total collected genres to avoid slowdown
        if len(result) >= 100:
            return
        
        for char, child_node in node.items():
            if char not in ('_end', '_tokens'):
                self._collect_complete_genres(child_node, prefix + char, result, max_depth - 1)
    
    def _precompute_char_token_mapping(self):
        """
        Precompute mapping from characters to token IDs and token decoded texts.
        This allows O(1) lookup instead of calling tokenizer.encode()/decode() at runtime.
        
        Time complexity: O(vocab_size) - runs once during initialization
        
        Note: Many subword tokenizers (like Qwen) add space prefixes to tokens.
        We need to handle both the raw first char and the first non-space char.
        """
        self._char_to_tokens: Dict[str, set] = {}
        self._token_to_text: Dict[int, str] = {}  # Precomputed decoded text for each token
        
        # For each token in vocabulary, get its decoded text
        for token_id in range(self.vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
                
                if not text:
                    continue
                
                # Store the decoded text (normalized to lowercase)
                # Keep leading spaces for proper concatenation (e.g., " rock" in "pop rock")
                # Only rstrip trailing whitespace, unless it's a pure whitespace token
                text_lower = text.lower()
                if text_lower.strip():  # Has non-whitespace content
                    normalized_text = text_lower.rstrip()
                else:  # Pure whitespace token
                    normalized_text = " "  # Normalize to single space
                self._token_to_text[token_id] = normalized_text
                
                # Map first character (including space) to this token
                first_char = text[0].lower()
                if first_char not in self._char_to_tokens:
                    self._char_to_tokens[first_char] = set()
                self._char_to_tokens[first_char].add(token_id)
                
                # Also map first non-space character to this token
                # This handles tokenizers that add space prefixes (e.g., " pop" -> maps to 'p')
                stripped_text = text.lstrip()
                if stripped_text and stripped_text != text:
                    first_nonspace_char = stripped_text[0].lower()
                    if first_nonspace_char not in self._char_to_tokens:
                        self._char_to_tokens[first_nonspace_char] = set()
                    self._char_to_tokens[first_nonspace_char].add(token_id)
                    
            except Exception:
                continue
        
        if self.debug:
            logger.debug(f"Precomputed char->token mapping for {len(self._char_to_tokens)} unique characters")
    
    def _try_reload_genres_vocab(self):
        """Check if genres vocab file has been updated and reload if necessary."""
        if not os.path.exists(self.genres_vocab_path):
            return
        
        try:
            mtime = os.path.getmtime(self.genres_vocab_path)
            if mtime > self.genres_vocab_mtime:
                self._load_genres_vocab()
        except Exception:
            pass  # Ignore errors during hot reload check
    
    def _get_genres_trie_node(self, prefix: str) -> Optional[Dict]:
        """
        Get the trie node for a given prefix.
        Returns None if the prefix is not valid (no genres start with this prefix).
        """
        node = self.genres_trie
        for char in prefix.lower():
            if char not in node:
                return None
            node = node[char]
        return node
    
    def _is_complete_genre(self, text: str) -> bool:
        """Check if the given text is a complete genre in the vocabulary."""
        node = self._get_genres_trie_node(text.strip())
        return node is not None and node.get('_end', False)
    
    def _get_trie_node_from_trie(self, trie: Dict, prefix: str) -> Optional[Dict]:
        """Get a trie node from a specific trie (helper for caption vs full trie)."""
        node = trie
        for char in prefix.lower():
            if char not in node:
                return None
            node = node[char]
        return node
    
    def _get_allowed_genres_tokens(self) -> List[int]:
        """
        Get allowed tokens for genres field based on trie matching.
        
        The entire genres string (including commas) must match a complete entry in the vocab.
        For example, if vocab contains "pop, rock, jazz", the generated string must exactly
        match that entry - we don't treat commas as separators for individual genres.
        
        Strategy:
        1. If caption-matched genres exist, use that smaller trie first (faster + more relevant)
        2. If no caption matches or prefix not in caption trie, fallback to full vocab trie
        3. Get valid next characters from current trie node
        4. For each candidate token, verify the full decoded text forms a valid trie prefix
        """
        if not self.genres_vocab:
            # No vocab loaded, allow all except newline if empty
            return []
        
        # Use the full accumulated value (don't split by comma - treat as single entry)
        accumulated = self.accumulated_value.lower()
        current_genre_prefix = accumulated.strip()
        
        # Determine which trie to use: caption-matched (priority) or full vocab (fallback)
        use_caption_trie = False
        current_node = None
        
        # Try caption-matched trie first if available
        if self.caption_genres_trie:
            if current_genre_prefix == "":
                current_node = self.caption_genres_trie
                use_caption_trie = True
            else:
                current_node = self._get_trie_node_from_trie(self.caption_genres_trie, current_genre_prefix)
                if current_node is not None:
                    use_caption_trie = True
        
        # Fallback to full vocab trie
        if current_node is None:
            if current_genre_prefix == "":
                current_node = self.genres_trie
            else:
                current_node = self._get_genres_trie_node(current_genre_prefix)
        
        if current_node is None:
            # Invalid prefix, force newline to end
            if self.newline_token:
                return [self.newline_token]
            return []
        
        # Get valid next characters from trie node
        valid_next_chars = set(k for k in current_node.keys() if k not in ('_end', '_tokens'))
        
        # If current value is a complete genre, allow newline to end
        is_complete = current_node.get('_end', False)
        
        if not valid_next_chars:
            # No more characters to match, only allow newline if complete
            allowed = set()
            if is_complete and self.newline_token:
                allowed.add(self.newline_token)
            return list(allowed)
        
        # Collect candidate tokens based on first character
        candidate_tokens = set()
        for char in valid_next_chars:
            if char in self._char_to_tokens:
                candidate_tokens.update(self._char_to_tokens[char])
        
        # Select the appropriate trie for validation
        active_trie = self.caption_genres_trie if use_caption_trie else self.genres_trie
        
        # Validate each candidate token: check if prefix + decoded_token is a valid trie prefix
        allowed = set()
        for token_id in candidate_tokens:
            # Use precomputed decoded text (already normalized)
            decoded_normalized = self._token_to_text.get(token_id, "")
            
            if not decoded_normalized or not decoded_normalized.strip():
                # Token decodes to empty or only whitespace - allow if space/comma is a valid next char
                if ' ' in valid_next_chars or ',' in valid_next_chars:
                    allowed.add(token_id)
                continue
            
            # Build new prefix by appending decoded token
            # Handle space-prefixed tokens (e.g., " rock" from "pop rock")
            if decoded_normalized.startswith(' ') or decoded_normalized.startswith(','):
                # Token has leading space/comma - append directly
                new_prefix = current_genre_prefix + decoded_normalized
            else:
                new_prefix = current_genre_prefix + decoded_normalized
            
            # Check if new_prefix is a valid prefix in the active trie
            new_node = self._get_trie_node_from_trie(active_trie, new_prefix)
            if new_node is not None:
                allowed.add(token_id)
        
        # If current value is a complete genre, also allow newline
        if is_complete and self.newline_token:
            allowed.add(self.newline_token)
        
        return list(allowed)
    
    def reset(self):
        """Reset the processor state for a new generation."""
        self.state = FSMState.THINK_TAG
        self.position_in_state = 0
        self.accumulated_value = ""  # Legacy, kept for compatibility
        self.accumulated_token_ids = []  # Reset token ID sequence
        self.codes_count = 0  # Reset codes counter
        self.user_field_token_queue = []  # Reset user field token queue
        self.current_user_field = None  # Reset current user field
    
    def set_target_duration(self, duration: Optional[float]):
        """
        Set the target duration for codes generation.
        
        Args:
            duration: Target duration in seconds. If None, no duration constraint is applied.
                     5 codes = 1 second, so target_codes = duration * 5.
        """
        self.target_duration = duration
        if duration is not None and duration > 0:
            self.target_codes = int(duration * 5)
            if self.debug:
                logger.debug(f"Set target duration: {duration}s -> {self.target_codes} codes")
        else:
            self.target_codes = None
            if self.debug:
                logger.debug("Target duration cleared, no duration constraint")
    
    def update_caption(self, caption: Optional[str]):
        """
        Update the caption and rebuild the caption-matched genres trie.
        Call this before each generation to prioritize genres from the new caption.
        
        Args:
            caption: User's input caption. If None or empty, clears caption matching.
        """
        # Check for hot reload of genres vocabulary
        self._try_reload_genres_vocab()
        
        self.caption = caption
        self.caption_genres_trie = {}
        self.caption_matched_genres = []
        
        if caption:
            self._extract_caption_genres(caption)
        
        # Also reset FSM state for new generation
        self.reset()
    
    def _get_allowed_tokens_for_fixed_string(self, fixed_str: str) -> List[int]:
        """
        Get the token IDs that can continue the fixed string from current position.
        Returns list of allowed token IDs.
        
        Strategy: Find the longest prefix that encodes to a single token, and return that token.
        This ensures we generate by tokens, not character-by-character.
        """
        remaining = fixed_str[self.position_in_state:]
        if not remaining:
            return []
        
        if self.debug:
            logger.debug(f"_get_allowed_tokens_for_fixed_string: fixed_str={repr(fixed_str)}, position_in_state={self.position_in_state}, remaining={repr(remaining)}")
        
        # Try encoding progressively longer prefixes, from longest to shortest
        # We want to find the longest prefix that encodes to a single token
        best_token = None
        best_prefix_len = 0
        
        # First pass: find the longest prefix that encodes to exactly one token
        for end in range(len(remaining), 0, -1):  # Start from longest prefix
            prefix = remaining[:end]
            tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            if tokens and len(tokens) == 1:
                # Found a prefix that encodes to a single token
                # Use this one (longest match)
                best_token = tokens[0]
                best_prefix_len = end
                if self.debug:
                    logger.debug(f"Found single-token match: prefix={repr(prefix)}, token_id={best_token}, token_text={repr(self.tokenizer.decode([best_token]))}")
                break
        
        # If we found a single-token match, return it (this is the preferred case)
        if best_token is not None:
            return [best_token]
        
        # Fallback: if no single-token match found, collect all possible first tokens
        # This handles edge cases where the string might need multiple tokens
        # But we still want to prefer longer matches
        # IMPORTANT: Only consider tokens that actually match the beginning of remaining string
        # Decode each candidate token and verify it matches the prefix
        allowed_tokens = {}
        for end in range(1, min(len(remaining) + 1, 20)):  # Limit search to avoid too many iterations
            prefix = remaining[:end]
            tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            if tokens:
                first_token = tokens[0]
                # Verify: decode the token and check it matches the prefix start
                decoded_token = self.tokenizer.decode([first_token])
                # Normalize both for comparison (strip and lower)
                normalized_prefix = prefix.lstrip().lower()
                normalized_decoded = decoded_token.lstrip().lower()
                
                # Check if decoded token matches the prefix start (allowing for space prefixes)
                if normalized_decoded.startswith(normalized_prefix) or normalized_prefix.startswith(normalized_decoded):
                    # Store the longest prefix length for each token
                    if first_token not in allowed_tokens or end > allowed_tokens[first_token]:
                        allowed_tokens[first_token] = end
        
        # Return tokens sorted by prefix length (longest first)
        # This ensures we prefer longer matches
        sorted_tokens = sorted(allowed_tokens.items(), key=lambda x: x[1], reverse=True)
        result = [token for token, _ in sorted_tokens] if sorted_tokens else []
        
        if self.debug:
            logger.debug(f"Fallback: returning {len(result)} tokens: {[(t, repr(self.tokenizer.decode([t]))) for t in result[:5]]}")
            if result:
                logger.debug(f"Fixed string: {repr(fixed_str)}, position: {self.position_in_state}, remaining: {repr(remaining)}")
        
        return result
    
    def _get_allowed_digit_tokens(self, min_val: int, max_val: int) -> List[int]:
        """
        Get allowed digit tokens based on accumulated value and range constraints.
        Uses early-blocking to prevent out-of-range values.
        """
        if not self.accumulated_value:
            # First digit: determine valid starting digits
            allowed_digits = set()
            for v in range(min_val, max_val + 1):
                allowed_digits.add(int(str(v)[0]))
            return [self.digit_tokens[d] for d in allowed_digits if d in self.digit_tokens]
        
        current = int(self.accumulated_value)
        allowed = []
        
        for d in range(10):
            new_value = int(self.accumulated_value + str(d))
            # Check if this digit could lead to a valid final value
            # A digit is valid if:
            # 1. new_value <= max_val (not already exceeded)
            # 2. new_value could potentially reach >= min_val
            #    (i.e., new_value * 10^k >= min_val for some k >= 0)
            
            if new_value > max_val:
                continue  # Already exceeded max
            
            # Check if we can still reach min_val
            # If new_value is already >= min_val, it's valid
            # If new_value < min_val, we need more digits, but new_value * 10 must not exceed max
            if new_value >= min_val:
                allowed.append(d)
            elif new_value * 10 <= max_val:
                # Can add more digits
                allowed.append(d)
        
        return [self.digit_tokens[d] for d in allowed if d in self.digit_tokens]
    
    def _get_allowed_numeric_tokens(self, prefix_tree: Dict[Tuple[int, ...], Set[int]]) -> List[int]:
        """
        Get allowed tokens for numeric field using the precomputed prefix tree.
        
        IMPORTANT: Uses token ID sequence as key (not string) to avoid tokenization mismatches.
        
        Args:
            prefix_tree: Precomputed prefix tree mapping token ID sequence -> set of allowed token IDs
            
        Returns:
            List of allowed token IDs for current accumulated_token_ids
        """
        token_prefix = tuple(self.accumulated_token_ids)
        
        if token_prefix in prefix_tree:
            return list(prefix_tree[token_prefix])
        
        # No valid continuation found - return empty list
        # The caller will handle this by forcing newline to end the field
        return []
    
    def _should_end_numeric_field(self, logits: torch.Tensor, min_val: int, max_val: int) -> bool:
        """
        Determine if we should end the current numeric field.
        Returns True if P(newline) > P(any valid digit) AND current value is valid.
        """
        if not self.accumulated_value:
            return False
        
        current = int(self.accumulated_value)
        if current < min_val or current > max_val:
            return False  # Can't end yet, value not in range
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        newline_prob = probs[0, self.newline_token].item() if self.newline_token else 0
        
        # Get max probability among valid digit tokens
        allowed_digits = self._get_allowed_digit_tokens(min_val, max_val)
        if not allowed_digits:
            return True  # No more digits possible, must end
        
        max_digit_prob = max(probs[0, t].item() for t in allowed_digits)
        
        if self.debug:
            logger.debug(f"Numeric field decision: newline_prob={newline_prob:.4f}, max_digit_prob={max_digit_prob:.4f}")
        
        return newline_prob > max_digit_prob
    
    def _should_end_text_field(self, logits: torch.Tensor) -> bool:
        """
        Determine if we should end a text field (genres).
        Returns True if P(newline) > P(any other token) AND we have some content.
        """
        if not self.accumulated_value.strip():
            return False  # Need at least some content
        
        probs = torch.softmax(logits, dim=-1)
        newline_prob = probs[0, self.newline_token].item() if self.newline_token else 0
        
        # Get max probability among non-newline tokens
        masked_probs = probs.clone()
        if self.newline_token:
            masked_probs[0, self.newline_token] = 0
        max_other_prob = masked_probs[0].max().item()
        
        return newline_prob > max_other_prob
    
    def _get_allowed_keyscale_tokens(self) -> List[int]:
        """
        Get allowed tokens for keyscale field using the precomputed prefix tree.
        Uses token ID sequence as key (not string) to avoid tokenization mismatches.
        """
        # Use token ID sequence as key
        token_prefix = tuple(self.accumulated_token_ids)
        
        if token_prefix in self.keyscale_prefix_tree:
            return list(self.keyscale_prefix_tree[token_prefix])
        
        # Fallback: if we somehow drifted off (shouldn't happen with constrained decoding),
        # return empty to force newline logic or stop.
        return []
    
    def _is_keyscale_complete(self) -> bool:
        """
        Check if keyscale value is complete and valid.
        Uses token ID sequence to check if current prefix allows newline.
        """
        token_prefix = tuple(self.accumulated_token_ids)
        # If current token sequence prefix is in tree and allows newline, it's complete
        if token_prefix in self.keyscale_prefix_tree:
            return self.newline_token in self.keyscale_prefix_tree[token_prefix]
        return False
    
    def _get_allowed_timesig_tokens(self) -> List[int]:
        """
        Get allowed tokens for timesignature field using the precomputed prefix tree.
        Uses token ID sequence as key (not string) to avoid tokenization mismatches.
        """
        token_prefix = tuple(self.accumulated_token_ids)
        
        if token_prefix in self.timesig_prefix_tree:
            return list(self.timesig_prefix_tree[token_prefix])
        
        # No valid continuation found - return empty list
        # The caller will handle this by forcing newline to end the field
        return []
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply constrained decoding by modifying logits.
        
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            scores: [batch_size, vocab_size] logits for next token
            
        Returns:
            Modified scores with invalid tokens masked to -inf and temperature scaling applied
        """
        if not self.enabled:
            return self._apply_temperature_scaling(scores)
        
        if self.state == FSMState.COMPLETED:
            return self._apply_temperature_scaling(scores)
        
        if self.state == FSMState.CODES_GENERATION:
            # Apply duration constraint in codes generation phase
            if self.target_codes is not None and self.eos_token_id is not None:
                if self.codes_count < self.target_codes:
                    # Block EOS token until target codes count is reached
                    scores[:, self.eos_token_id] = float('-inf')
                    if self.debug:
                        logger.debug(f"Codes generation: {self.codes_count}/{self.target_codes}, blocking EOS")
                else:
                    # Force EOS token when target codes count is reached
                    mask = torch.full_like(scores, float('-inf'))
                    mask[:, self.eos_token_id] = 0
                    scores = scores + mask
                    if self.debug:
                        logger.debug(f"Codes generation: {self.codes_count}/{self.target_codes}, forcing EOS")
            return self._apply_temperature_scaling(scores)
        
        batch_size = scores.shape[0]
        
        # Process each sequence in batch
        for b in range(batch_size):
            result = self._process_single_sequence(input_ids[b], scores[b:b+1])
            scores[b] = result[0]  # result is [1, vocab_size], need [vocab_size]
        
        # Apply temperature scaling after constraint masking
        return self._apply_temperature_scaling(scores)
    
    def _apply_temperature_scaling(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply temperature scaling based on current generation phase.
        
        Temperature scaling: logits = logits / temperature
        - Lower temperature (< 1.0) makes distribution sharper (more deterministic)
        - Higher temperature (> 1.0) makes distribution flatter (more diverse)
        
        Args:
            scores: [batch_size, vocab_size] logits
            
        Returns:
            Temperature-scaled logits
        """
        # Determine which temperature to use based on current state
        if self.state == FSMState.CODES_GENERATION or self.state == FSMState.COMPLETED:
            temperature = self.codes_temperature
        else:
            temperature = self.metadata_temperature
        
        # If no temperature is set for this phase, return scores unchanged
        if temperature is None:
            return scores
        
        # Avoid division by zero
        if temperature <= 0:
            temperature = 1e-6
        
        # Apply temperature scaling
        return scores / temperature
    
    def _get_user_provided_field_tokens(self, field_name: str) -> Optional[List[int]]:
        """
        Get token sequence for a user-provided field (field_name + value + newline).
        Uses the same tokenization logic as prefix tree building.
        
        Args:
            field_name: Field name ("bpm", "duration", "keyscale", "timesignature", "genres")
            
        Returns:
            List of token IDs for the complete field, or None if field is not provided
        """
        value = self.user_provided_metadata.get(field_name)
        if value is None:
            return None
        
        # Build full field string with space (matching prefix tree tokenization)
        field_to_prefix = {
            "bpm": "bpm: ",
            "duration": "duration: ",
            "keyscale": "keyscale: ",
            "timesignature": "timesignature: ",
            "genres": "genres: ",
        }
        prefix = field_to_prefix[field_name]
        full_text = f"{prefix}{value}\n"
        
        # Tokenize the full field
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # Extract only the field tokens (skip the prefix tokens that match state machine output)
        # The state machine generates "field_name:" (no space), so we need to match that
        prefix_for_matching = field_name + ":"
        prefix_tokens = self.tokenizer.encode(prefix_for_matching, add_special_tokens=False)
        
        # Find where prefix ends in full tokens
        if len(tokens) >= len(prefix_tokens) and tokens[:len(prefix_tokens)] == prefix_tokens:
            # Return tokens after prefix (field value + newline)
            return tokens[len(prefix_tokens):]
        else:
            # Fallback: return all tokens (shouldn't happen if tokenization is consistent)
            if self.debug:
                logger.warning(f"Could not match prefix tokens for field {field_name}, using all tokens")
            return tokens
    
    def _process_single_sequence(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Process a single sequence and return modified scores."""
        
        # Check if we have tokens in queue for user-provided field
        # If so, inject the next token directly
        if self.user_field_token_queue:
            mask = torch.full_like(scores, float('-inf'))
            next_token = self.user_field_token_queue[0]
            mask[0, next_token] = 0
            scores = scores + mask
            return scores
        
        # Create mask (all -inf initially)
        mask = torch.full_like(scores, float('-inf'))
        
        if self.state in self.fixed_strings:
            # Fixed string state: force specific tokens
            fixed_str = self.fixed_strings[self.state]
            allowed = self._get_allowed_tokens_for_fixed_string(fixed_str)
            
            if allowed:
                # Check if we should stop at reasoning (after </think> tag)
                # This happens when we're about to complete the </think> tag
                if self.state == FSMState.THINK_END_TAG and self.stop_at_reasoning:
                    # Check if the next token would complete the fixed string
                    # We check if position_in_state + length of next token would complete it
                    # Since we don't know which token will be selected, we check if we're close to completion
                    # Actually, a better approach: check if this is the last character(s) of the fixed string
                    remaining_chars = len(fixed_str) - self.position_in_state
                    # If remaining is small (<= 10 chars, which is typically 1-2 tokens), force EOS
                    if remaining_chars <= 10:
                        # Force EOS token to stop generation
                        if self.eos_token_id is not None:
                            mask[0, self.eos_token_id] = 0
                            scores = scores + mask
                            if self.debug:
                                logger.debug(f"stop_at_reasoning=True: forcing EOS near end of </think> tag (remaining: {remaining_chars} chars)")
                            return scores
                
                for t in allowed:
                    mask[0, t] = 0
                # Apply mask
                scores = scores + mask
                
                # Update position tracking
                # We need to check if the selected token completes the fixed string
                # This will be done in update_state() after token selection
            else:
                # Position exceeds string, move to next state
                # If stop_at_reasoning is True and we're transitioning from THINK_END_TAG,
                # force EOS before transitioning
                if self.state == FSMState.THINK_END_TAG and self.stop_at_reasoning:
                    # Force EOS token to stop generation
                    if self.eos_token_id is not None:
                        mask[0, self.eos_token_id] = 0
                        scores = scores + mask
                        if self.debug:
                            logger.debug(f"stop_at_reasoning=True: forcing EOS after completing </think> tag")
                        return scores
                
                old_state = self.state
                self._transition_to_next_state()
                # Avoid infinite recursion: if we're still in a fixed_strings state, just return scores
                if self.state in self.fixed_strings:
                    # This shouldn't happen, but if it does, just return scores to avoid recursion
                    if self.debug:
                        logger.warning(f"State transition from {old_state.name} to {self.state.name} still in fixed_strings, avoiding recursion")
                    return scores
                return self._process_single_sequence(input_ids, torch.zeros_like(scores))
        
        elif self.state == FSMState.BPM_VALUE:
            # Check if field is user-provided and we haven't started injecting yet
            if self.user_provided_metadata["bpm"] is not None and not self.user_field_token_queue and not self.accumulated_token_ids:
                # Initialize token queue with field value tokens (value + newline)
                value = self.user_provided_metadata["bpm"]
                # Tokenize " value\n" (space + value + newline) to match actual tokenization
                value_text = f" {value}\n"
                value_tokens = self.tokenizer.encode(value_text, add_special_tokens=False)
                if value_tokens:
                    self.user_field_token_queue = value_tokens
                    self.current_user_field = "bpm"
                    # Inject first token
                    mask[0, value_tokens[0]] = 0
                    scores = scores + mask
                    return scores
            
            # Allow valid numeric tokens using prefix tree (supports multi-digit tokens like "120")
            allowed = self._get_allowed_numeric_tokens(self.bpm_prefix_tree)
            for t in allowed:
                mask[0, t] = 0
            
            # Also allow newline if current token sequence prefix allows it
            # Check if current token sequence is in prefix tree and allows newline
            token_prefix = tuple(self.accumulated_token_ids)
            if token_prefix in self.bpm_prefix_tree and self.newline_token in self.bpm_prefix_tree[token_prefix]:
                mask[0, self.newline_token] = 0
            
            scores = scores + mask
        
        elif self.state == FSMState.DURATION_VALUE:
            # Check if field is user-provided and we haven't started injecting yet
            if self.user_provided_metadata["duration"] is not None and not self.user_field_token_queue and not self.accumulated_token_ids:
                # Initialize token queue with field value tokens (value + newline)
                value = self.user_provided_metadata["duration"]
                value_text = f" {value}\n"
                value_tokens = self.tokenizer.encode(value_text, add_special_tokens=False)
                if value_tokens:
                    self.user_field_token_queue = value_tokens
                    self.current_user_field = "duration"
                    # Inject first token
                    mask[0, value_tokens[0]] = 0
                    scores = scores + mask
                    return scores
            
            # If target_duration is set, force generate that exact value
            if self.target_duration is not None:
                target_str = str(int(self.target_duration))
                current_pos = len(self.accumulated_value)
                
                if current_pos < len(target_str):
                    # Force the next digit
                    next_digit = int(target_str[current_pos])
                    if next_digit in self.digit_tokens:
                        mask[0, self.digit_tokens[next_digit]] = 0
                else:
                    # All digits generated, force newline
                    if self.newline_token:
                        mask[0, self.newline_token] = 0
                
                scores = scores + mask
            else:
                # Normal duration generation with range constraint
                # Allow valid numeric tokens using prefix tree (supports multi-digit tokens like "60", "120")
                allowed = self._get_allowed_numeric_tokens(self.duration_prefix_tree)
                for t in allowed:
                    mask[0, t] = 0
                
                # Also allow newline if current token sequence prefix allows it
                token_prefix = tuple(self.accumulated_token_ids)
                if token_prefix in self.duration_prefix_tree and self.newline_token in self.duration_prefix_tree[token_prefix]:
                    mask[0, self.newline_token] = 0
                
                scores = scores + mask
        
        elif self.state == FSMState.GENRES_VALUE:
            # Check if field is user-provided and we haven't started injecting yet
            if self.user_provided_metadata["genres"] is not None and not self.user_field_token_queue and not self.accumulated_value:
                # Initialize token queue with field value tokens (value + newline)
                value = self.user_provided_metadata["genres"]
                value_text = f" {value}\n"
                value_tokens = self.tokenizer.encode(value_text, add_special_tokens=False)
                if value_tokens:
                    self.user_field_token_queue = value_tokens
                    self.current_user_field = "genres"
                    # Inject first token
                    mask[0, value_tokens[0]] = 0
                    scores = scores + mask
                    return scores
            
            # Try to hot-reload genres vocab if file has changed
            self._try_reload_genres_vocab()
            
            # Get allowed tokens based on genres vocabulary
            allowed = self._get_allowed_genres_tokens()
            
            if allowed:
                # Use vocabulary-constrained decoding
                for t in allowed:
                    mask[0, t] = 0
                scores = scores + mask
            elif self.genres_vocab:
                # Vocab is loaded but no valid continuation found
                # Force newline to end the field
                if self.newline_token:
                    mask[0, self.newline_token] = 0
                    if self.debug:
                        logger.debug(f"No valid genre continuation for '{self.accumulated_value}', forcing newline")
                scores = scores + mask
            else:
                # Fallback: no vocab loaded, use probability-based ending
                if self._should_end_text_field(scores):
                    if self.newline_token:
                        mask[0, self.newline_token] = 0
                        self._transition_to_next_state()
                    scores = scores + mask
                else:
                    # Allow any token except newline if we don't have content yet
                    if not self.accumulated_value.strip():
                        if self.newline_token:
                            scores[0, self.newline_token] = float('-inf')
                    # Otherwise, don't constrain (fallback behavior)
        
        elif self.state == FSMState.KEYSCALE_VALUE:
            # Check if field is user-provided and we haven't started injecting yet
            if self.user_provided_metadata["keyscale"] is not None and not self.user_field_token_queue and not self.accumulated_token_ids:
                # Initialize token queue with field value tokens (value + newline)
                value = self.user_provided_metadata["keyscale"]
                value_text = f" {value}\n"
                value_tokens = self.tokenizer.encode(value_text, add_special_tokens=False)
                if value_tokens:
                    self.user_field_token_queue = value_tokens
                    self.current_user_field = "keyscale"
                    # Inject first token
                    mask[0, value_tokens[0]] = 0
                    scores = scores + mask
                    return scores
            
            # Check if current token sequence is complete (allows newline)
            token_prefix = tuple(self.accumulated_token_ids)
            if token_prefix in self.keyscale_prefix_tree and self.newline_token in self.keyscale_prefix_tree[token_prefix]:
                # Complete keyscale, allow newline
                if self.newline_token:
                    mask[0, self.newline_token] = 0
                scores = scores + mask
            else:
                # Not complete, allow valid continuation tokens
                allowed = self._get_allowed_keyscale_tokens()
                if allowed:
                    for t in allowed:
                        mask[0, t] = 0
                    scores = scores + mask
                else:
                    # No valid tokens found - force newline to end field
                    # This handles edge cases where keyscale format is unexpected
                    if self.newline_token:
                        mask[0, self.newline_token] = 0
                    scores = scores + mask
        
        elif self.state == FSMState.TIMESIG_VALUE:
            # Check if field is user-provided and we haven't started injecting yet
            if self.user_provided_metadata["timesignature"] is not None and not self.user_field_token_queue and not self.accumulated_token_ids:
                # Initialize token queue with field value tokens (value + newline)
                value = self.user_provided_metadata["timesignature"]
                value_text = f" {value}\n"
                value_tokens = self.tokenizer.encode(value_text, add_special_tokens=False)
                if value_tokens:
                    self.user_field_token_queue = value_tokens
                    self.current_user_field = "timesignature"
                    # Inject first token
                    mask[0, value_tokens[0]] = 0
                    scores = scores + mask
                    return scores
            
            # Check if current token sequence is complete (allows newline)
            token_prefix = tuple(self.accumulated_token_ids)
            if token_prefix in self.timesig_prefix_tree and self.newline_token in self.timesig_prefix_tree[token_prefix]:
                # Complete value, allow newline
                if self.newline_token:
                    mask[0, self.newline_token] = 0
                scores = scores + mask
            else:
                # Not complete, allow valid continuation tokens
                allowed = self._get_allowed_timesig_tokens()
                for t in allowed:
                    mask[0, t] = 0
                scores = scores + mask
        
        return scores
    
    def _transition_to_next_state(self):
        """Transition to the next FSM state."""
        if self.state in self.next_state:
            old_state = self.state
            next_state = self.next_state[self.state]
            
            # If stop_at_reasoning is True and we're transitioning from THINK_END_TAG,
            # skip CODES_GENERATION and go directly to COMPLETED
            if self.stop_at_reasoning and old_state == FSMState.THINK_END_TAG:
                next_state = FSMState.COMPLETED
                if self.debug:
                    logger.debug(f"stop_at_reasoning=True: skipping CODES_GENERATION, going directly to COMPLETED")
            
            self.state = next_state
            self.position_in_state = 0
            self.accumulated_value = ""  # Legacy, kept for compatibility
            self.accumulated_token_ids = []  # Reset token ID sequence for new field
            if self.debug:
                logger.debug(f"FSM transition: {old_state.name} -> {self.state.name}")
    
    def update_state(self, generated_token_id: int):
        """
        Update internal state after a token has been generated.
        This should be called after each token generation.
        
        Args:
            generated_token_id: The token ID that was just generated
        """
        if not self.enabled:
            return
        
        if self.state == FSMState.COMPLETED:
            return
        
        if self.state == FSMState.CODES_GENERATION:
            # Count generated codes for duration constraint
            self.codes_count += 1
            if self.debug and self.target_codes is not None:
                logger.debug(f"Codes count: {self.codes_count}/{self.target_codes}")
            return
        
        # Handle user-provided field token injection
        if self.user_field_token_queue:
            # Verify the generated token matches the expected token from queue
            expected_token = self.user_field_token_queue[0]
            if generated_token_id != expected_token:
                if self.debug:
                    logger.warning(f"Expected token {expected_token} but got {generated_token_id} for user-provided field {self.current_user_field}")
            
            # Remove consumed token from queue
            self.user_field_token_queue.pop(0)
            
            # If queue is empty, field injection is complete, transition to next state
            if not self.user_field_token_queue:
                if self.debug:
                    logger.debug(f"Completed injection of user-provided field: {self.current_user_field}")
                field_name = self.current_user_field
                self.current_user_field = None
                
                # Transition to next state (skip VALUE state since we already injected everything)
                # The next state should be determined by _get_next_field_state
                next_state = self._get_next_field_state(field_name)
                if next_state:
                    old_state = self.state
                    self.state = next_state
                    self.position_in_state = 0
                    self.accumulated_value = ""
                    self.accumulated_token_ids = []
                    if self.debug:
                        logger.debug(f"FSM transition (after user field injection): {old_state.name} -> {self.state.name}")
                else:
                    # All fields done, go to THINK_END_TAG
                    self._transition_to_next_state()
            return
        
        token_str = self.tokenizer.decode([generated_token_id])
        
        if self.debug:
            logger.debug(f"Generated token: {repr(token_str)} (id={generated_token_id}), state={self.state.name}")
        
        if self.state in self.fixed_strings:
            # Update position in fixed string
            fixed_str = self.fixed_strings[self.state]
            self.position_in_state += len(token_str)
            
            # Check if we've completed the fixed string
            if self.position_in_state >= len(fixed_str):
                self._transition_to_next_state()
        
        elif self.state in [FSMState.BPM_VALUE, FSMState.DURATION_VALUE, FSMState.TIMESIG_VALUE]:
            # Accumulate numeric value using token ID sequence
            if generated_token_id == self.newline_token:
                # if self.state == FSMState.DURATION_VALUE and self.accumulated_value:
                #     try:
                #         generated_duration = int(self.accumulated_value)
                #         if self.target_codes is None and generated_duration > 0:
                #             self.target_codes = int(generated_duration * 5)
                #             if self.debug:
                #                 logger.debug(f"Synced duration: {generated_duration}s -> Set target_codes limit to {self.target_codes}")
                #     except ValueError:
                #         if self.debug:
                #             logger.warning(f"Could not parse duration value: {self.accumulated_value}")
                # Newline ends the field
                # Save old state before transition
                old_state = self.state
                self._transition_to_next_state()
                # IMPORTANT: After state transition, if new state is a fixed_strings state,
                # we should NOT update position_in_state with the newline token length,
                # because that token belongs to the old state, not the new state.
                # Return early to avoid the fixed_strings update logic below.
                if self.state in self.fixed_strings:
                    return
            else:
                # Add token ID to sequence (for prefix tree lookup)
                self.accumulated_token_ids.append(generated_token_id)
                # Also update legacy accumulated_value for compatibility
                if token_str.strip().isdigit():
                    self.accumulated_value += token_str.strip()
        
        elif self.state == FSMState.GENRES_VALUE:
            if generated_token_id == self.newline_token:
                # Newline ends the field
                self._transition_to_next_state()
                # IMPORTANT: After state transition, if new state is a fixed_strings state,
                # we should NOT update position_in_state with the newline token length,
                # because that token belongs to the old state, not the new state.
                # Return early to avoid the fixed_strings update logic below.
                if self.state in self.fixed_strings:
                    return
            else:
                # Genres still uses string-based trie, so keep accumulated_value
                self.accumulated_value += token_str
        
        elif self.state == FSMState.KEYSCALE_VALUE:
            if generated_token_id == self.newline_token:
                # Newline ends the field
                self._transition_to_next_state()
                # IMPORTANT: After state transition, if new state is a fixed_strings state,
                # we should NOT update position_in_state with the newline token length,
                # because that token belongs to the old state, not the new state.
                # Return early to avoid the fixed_strings update logic below.
                if self.state in self.fixed_strings:
                    return
            else:
                # Add token ID to sequence (for prefix tree lookup)
                self.accumulated_token_ids.append(generated_token_id)
                # Also update legacy accumulated_value for compatibility
                self.accumulated_value += token_str

