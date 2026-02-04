# REAPER SCAN RESULTS - ACE-Step-1.5

## Scan Date: 2026-02-04
## Budget: MAX
## Reapers Deployed: 9

---

## SCOUT REPORTS

### Scout Alpha - Codebase Structure
- **Total Python Files**: 33 core files (~14,823 lines)
- **Architecture**: Hybrid LM + DiT music generation
- **Entry Points**: acestep (Gradio UI), acestep-api (FastAPI), acestep-download (CLI)
- **Key Modules**: handler.py (3,424 LOC), llm_inference.py (2,429 LOC), api_server.py (2,345 LOC)
- **No Test Suite**: Zero automated tests exist

### Scout Beta - Hot Spots
- **God Objects**: AceStepHandler (68 methods), LLMHandler (37+ methods)
- **Deep Nesting**: 3,516 occurrences of 20+ space indentation
- **Technical Debt Score**: 8/10 (estimated 2-3 months to address)

---

## INVESTIGATOR FINDINGS

### Runtime Bugs (bug-hunter)
| ID | Severity | File | Line | Issue |
|----|----------|------|------|-------|
| BUG-003 | HIGH | handler.py | 1614 | Unguarded None dereference on target_wavs |
| BUG-005 | HIGH | inference.py | 371 | Unguarded llm_handler access |
| BUG-006 | MEDIUM | api_server.py | 626-646 | Race condition in _JobStore mark_* methods |
| BUG-009 | MEDIUM | api_server.py | 132-140 | Socket resource leak in _can_access_google |
| BUG-011 | MEDIUM | handler.py | 2879 | torch.cuda.empty_cache without availability check |
| BUG-001 | MEDIUM | inference.py | 616 | TypeError if seed_list is None |

### Security Issues (security-hardener)
| ID | Severity | File | Line | Issue |
|----|----------|------|------|-------|
| SEC-001 | HIGH | openrouter/openrouter_api_server.py | 90 | Timing-unsafe API key comparison |
| SEC-002 | HIGH | api_server.py | 1869 | No path validation on ref_audio_path/src_audio_path |
| SEC-003 | HIGH | api_server.py | 1370 | No enforcement of GPU-tier resource limits (DoS) |
| SEC-004 | MEDIUM | third_parts/nano-vllm | - | Pickle deserialization in shared memory IPC |
| SEC-005 | MEDIUM | api_server.py | - | No rate limiting on API endpoints |

### Code Quality (strict-code-reviewer)
| ID | Severity | Files | Issue |
|----|----------|-------|-------|
| QUAL-001 | MEDIUM | api_server.py, handler.py, api_routes.py | Duplicate _get_project_root() |
| QUAL-002 | MEDIUM | handler.py, llm_inference.py, api_server.py | Significant unused imports |
| QUAL-005 | MEDIUM | inference.py:254 | Return type annotation mismatch (says 4, returns 7) |
| QUAL-008 | MEDIUM | llm_inference.py:355 | 80-90s tokenizer loading delay (known issue) |

### Architecture (principal-architect)
| ID | Priority | Issue |
|----|----------|-------|
| ARCH-001 | HIGH | Duplicated path validation logic (api_server.py, api_routes.py) |
| ARCH-002 | HIGH | Two separate API implementations with different codepaths |
| ARCH-003 | MEDIUM | Handler classes mix concerns (god objects) |
| ARCH-004 | MEDIUM | Global state via module-level variables |
| ARCH-007 | MEDIUM | No unified request/response models |

### Test Coverage (test-strategist)
- **ZERO automated tests exist**
- Critical gaps: API security, audio processing, FSM constrained decoding, LoRA loading

### Simplification (code-simplifier)
| ID | Priority | Lines Saved | Issue |
|----|----------|-------------|-------|
| SIMP-002 | HIGH | ~45 | Repeated metadata parsing logic (3 places) |
| SIMP-006 | MEDIUM | ~35 | Complex nested conditionals in prepare_padding_info |
| SIMP-003 | MEDIUM | ~40 | Over-defensive device checking code |
| SIMP-001 | MEDIUM | ~25 | Duplicated audio normalization code |

---

## VERIFIED EXECUTION QUEUE

### CRITICAL (Security)
1. SEC-001: Fix timing-unsafe API key comparison in OpenRouter
2. SEC-002: Add path validation for ref_audio_path/src_audio_path
3. SEC-003: Enforce GPU-tier resource limits

### HIGH (Bugs)
4. BUG-003: Guard None dereference on target_wavs
5. BUG-005: Guard llm_handler access in inference.py

### HIGH (Architecture)
6. ARCH-001: Extract shared security/auth code to utils module
7. QUAL-001: Consolidate duplicate _get_project_root()

### MEDIUM (Bugs)
8. BUG-006: Handle missing job_id in _JobStore methods
9. BUG-009: Fix socket leak in _can_access_google
10. BUG-011: Guard torch.cuda.empty_cache calls

### MEDIUM (Code Quality)
11. QUAL-002: Remove unused imports
12. QUAL-005: Fix return type annotation in inference.py
13. SIMP-002: Consolidate metadata parsing logic

### LOW (Already Verified from Previous Session)
14. H6: Replace hardcoded token ID 151643 with tokenizer.eos_token_id

---

## SUMMARY

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Security | 3 | 0 | 2 | 0 | 5 |
| Bugs | 0 | 2 | 4 | 0 | 6 |
| Quality | 0 | 2 | 4 | 0 | 6 |
| Total | 3 | 4 | 10 | 1 | 18 |

**Estimated Effort**: 2-3 days for critical/high, 1 week for medium
