1. Add an activation hook to extract residual state with less storage overhead. 
2. For full fidelity, implement windowing instead of truncation in the GPT2 encoder [Redundant, changed max length back to 1024]
3. add decoder normalization during training

## SAE parity with Bricken et al. (2023): To-dos (compute-aware)

### P0 — Must-do (highest impact for interpretability)
- [ ] **Add pre-encoder biasing:** encode on `(x - b_dec)` (subtract decoder bias before the encoder); keep both encoder + decoder biases.
- [ ] **Match loss scaling exactly:** reconstruction **MSE = mean over input dims**; sparsity **L1 = sum over hidden units** (then average over batch). Write the exact formula in docs/comments so it doesn’t drift.
- [ ] **Define interpretability KPIs (not just recon loss):**
  - [ ] average active features per token/vector
  - [ ] firing-rate histogram across features
  - [ ] dead-feature fraction (never/rarely fires)
  - [ ] stability of top examples per feature across checkpoints

### P1 — Strongly recommended (prevents collapse, improves feature quality)
- [ ] **Add dictionary weight normalization:** enforce unit norms for dictionary vectors (pick a convention, typically decoder columns) on a schedule (e.g., every N steps).
- [ ] **Implement dead-feature resampling:** track features that haven’t fired in a sliding window; periodically reinit/resample them using high-reconstruction-error samples; reset optimizer state for affected params if feasible.
- [ ] **Set explicit sparsity targets:** choose a target average number of active features per token/vector and tune `λ` to hit it (avoid tuning purely for recon loss).

### P2 — Data/training pipeline (scaled-down for limited hardware)
- [ ] **Make activation caching a first-class dataset:**
  - [ ] cache activations to disk in chunks (memmap/chunked arrays) to avoid re-running the base model
  - [ ] store minimal metadata: sample id, token idx, layer, dtype/scale
- [ ] **Approximate “sampling without replacement”:** shuffle a large cached activation set and iterate once before reshuffling (avoid over-epoching a tiny cache).
- [ ] **Use reduced precision storage:** store activations in fp16/bf16 to fit more data; keep losses/accumulators in fp32.
- [ ] **Prefer step-based training over epoch-based:** optimize for many update steps on a large cache rather than many epochs on a small cache.

### P3 — Context window: document boundary leakage
- [ ] **Fix context window crossing document boundaries:** In `extract_activations.py`, tokens from multiple documents are concatenated into a flat array with no boundary markers. The ±10 token context window in `interpretability.py` (`process_chunk_batched`, lines 113–117) slides over this flat array, so tokens near document boundaries can display context mixing two unrelated documents. Low impact for now (activations are correct; boundary tokens are a small fraction of OpenWebText data), but when needed: store sequence lengths or boundary indices alongside `token_ids` when saving chunks, then clamp the context window to stay within the same document.

### P3 — `get_max_activating_features` efficiency
- [ ] **Use `heapq.nlargest`/`nsmallest` instead of full sort:** current implementation does O(n log n) sort per sample; switching to `heapq.nlargest` would give O(n log k), better when `top_k << active features`. Low priority since features are sparse post-threshold so `n` is already small.

### P3 — Lightweight interpretability checks (cheap, high signal)
- [ ] **Top-K contexts per feature report:** maintain rolling top-K activations with text snippets for each feature; regenerate at checkpoints.
- [ ] **Ablation sanity checks (activation space):** set selected feature activations to zero and measure:
  - [ ] change in reconstruction error
  - [ ] change in next-layer activation proxy (if available/cached)
- [ ] **Pinning sanity checks (activation space):** clamp selected features on and verify reconstruction shifts consistently along that feature’s decoder direction.
- [ ] **Monosemanticity heuristics:** cluster/keyword-score top-K contexts per feature to estimate “topic purity” and feature drift.

### P4 — Repo hygiene
- [ ] **Purge wandb artifacts from git history:** use `git filter-repo --path wandb/ --path notebooks/wandb/ --invert-paths` then force-push. All collaborators will need to re-clone afterward.

### P4 — Practical solo-research hygiene
- [ ] **Reproducible run config:** seeds, cache version/hash, SAE hyperparams, loss scaling, normalization schedule, resampling frequency.
- [ ] **Checkpoint interpretability artifacts:** save alongside weights:
  - [ ] firing stats + dead-feature list
  - [ ] top-K contexts per feature
  - [ ] small markdown “feature gallery” report
- [ ] **Compute-aware sweep plan:**
  - [ ] short pilot runs to validate loss scaling + sparsity targets
  - [ ] fewer, longer runs on the largest cache you can store
  - [ ] change one major factor at a time (λ, width, norm schedule, resampling cadence)

### Suggested execution order (if time/compute constrained)
1. [ ] Pre-encoder bias + correct loss scaling
2. [ ] Logging KPIs: firing rates, dead features, top-K contexts
3. [ ] Dictionary normalization
4. [ ] Dead-feature resampling
5. [ ] Large cached activation streaming + step-based training
6. [ ] Ablation/pinning checks + monosemanticity heuristics
