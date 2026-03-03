# TODO / Future Improvements

## Interpretability

### Context Window: Document Boundary Leakage
**File:** `src/evaluation/interpretability.py` — `process_chunk_batched`, lines 113–117

In `extract_activations.py`, tokens from multiple documents are concatenated into a flat array with no boundary markers (attention mask strips padding, then everything is `torch.cat`'d). The ±10 token context window in `interpretability.py` slides over this flat array, so tokens near document boundaries can display context that mixes two unrelated documents.

**Impact:** Low for current exploratory use — activations are correct (GPT-2 attention mask respects boundaries), and boundary tokens are a small fraction of OpenWebText data. A few garbled context windows won't derail feature interpretation.

**Fix when needed:** Preserve document boundary information when saving chunks (e.g., store sequence lengths or boundary indices alongside token_ids), then clamp the context window to stay within the same document.
