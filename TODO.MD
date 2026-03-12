1. Add an activation hook to extract residual state with less storage overhead.
2. For full fidelity, implement windowing instead of truncation in the GPT2 encoder [Redundant, changed max length back to 1024]
3. add decoder normalization during training

## SAE parity with Bricken et al. (2023): To-dos (compute-aware)

### P0 — Must-do (highest impact for interpretability)

- **Add pre-encoder biasing:** encode on `(x - b_dec)` (subtract decoder bias before the encoder); keep both encoder + decoder biases.
- **Match loss scaling exactly:** reconstruction **MSE = mean over input dims**; sparsity **L1 = sum over hidden units** (then average over batch). Write the exact formula in docs/comments so it doesn’t drift.
- **Define interpretability KPIs (not just recon loss):**
  - average active features per token/vector
  - firing-rate histogram across features
  - dead-feature fraction (never/rarely fires)
  - stability of top examples per feature across checkpoints

### P1 — Strongly recommended (prevents collapse, improves feature quality)

- **Fix initialization to reduce early feature death (P1-INIT):**
  - **Finding (2026-03-10):** Dead neuron proportion scales steeply with expansion factor (4× → 60%, 8× → 79%, 16× → 89%, 32× → 95%). Root cause is a combination of initialization and early training dynamics.
  - **Normalize `W_dec` at init:** `trainer.py` normalizes decoder columns every step (line 59), but `sparse_autoencoder.py` initializes `W_dec` with `randn/sqrt(d_hidden)` unnormalized — the very first training step operates against an unnormalized decoder. Apply `F.normalize(W_dec, dim=1)` at init.
  - **Initialize `b_enc` to small positive values (e.g., 0.01–0.1):** Currently `b_enc = zeros`. At init, ~50% of features have negative pre-activations; L1 immediately pushes those biases further negative, permanently killing them before they learn. A small positive init gives all features a fighting chance to fire early and find useful directions.
  - **Consider Kaiming uniform init for `W_enc`:** Current `randn/sqrt(d_model)` is close to Kaiming but uses normal distribution. `nn.init.kaiming_uniform_(W_enc, nonlinearity=’relu’)` is the standard choice for ReLU encoders.
- **Add λ (L1 penalty) warmup (P1-WARMUP):**
  - Currently `_lambda=0.01` is applied at full strength from step 1. Features that don’t fire in early batches get their biases pushed negative by L1 and never recover.
  - Linearly ramp λ from 0 to target over the first N steps (e.g., 1000–5000 steps). This gives features time to find useful directions before sparsity pressure kills them.
  - Implementation: pass `warmup_steps` to `SAETrainer`, compute `effective_lambda = _lambda * min(1, step / warmup_steps)` in the training loop.
- **Add learning rate warmup (P1-LR-WARMUP):**
  - Large early gradients can kill features in the first few steps. Use a linear LR warmup (e.g., 500–1000 steps) via `torch.optim.lr_scheduler.LinearLR` or manual schedule.
- **Add dictionary weight normalization:** enforce unit norms for dictionary vectors (pick a convention, typically decoder columns) on a schedule (e.g., every N steps).
- **Implement dead-feature resampling:** track features that haven’t fired in a sliding window; periodically reinit/resample them using high-reconstruction-error samples; reset optimizer state for affected params if feasible.
  - **Finding (2026-03-09):** 4x SAE has 1840/3072 dead features (~60%) across 173M token activations — effective expansion is ~1.6x, not 4x. Cause: dying ReLU + L1 penalty with no ghost grads or resampling. Fix options: dead-feature resampling (this item), or switch to TopK activation (forces exactly K features to fire per token, preventing permanent silence).
  - **Updated finding (2026-03-10):** Problem confirmed across all expansion factors. Dead-feature resampling is critical — without it, init + L1 dynamics kill the majority of features permanently. Resampling every ~25k–50k steps using the top reconstruction-error samples is a good starting cadence.
- **Set explicit sparsity targets:** choose a target average number of active features per token/vector and tune `λ` to hit it (avoid tuning purely for recon loss).

### P2 — Data/training pipeline (scaled-down for limited hardware)

- **Make activation caching a first-class dataset:**
  - cache activations to disk in chunks (memmap/chunked arrays) to avoid re-running the base model
  - store minimal metadata: sample id, token idx, layer, dtype/scale
- **Approximate “sampling without replacement”:** shuffle a large cached activation set and iterate once before reshuffling (avoid over-epoching a tiny cache).
- **Use reduced precision storage:** store activations in fp16/bf16 to fit more data; keep losses/accumulators in fp32.
- **Prefer step-based training over epoch-based:** optimize for many update steps on a large cache rather than many epochs on a small cache.

### P3 — Context window: document boundary leakage

- **Fix context window crossing document boundaries:** In `extract_activations.py`, tokens from multiple documents are concatenated into a flat array with no boundary markers. The ±10 token context window in `interpretability.py` (`process_chunk_batched`, lines 113–117) slides over this flat array, so tokens near document boundaries can display context mixing two unrelated documents. Low impact for now (activations are correct; boundary tokens are a small fraction of OpenWebText data), but when needed: store sequence lengths or boundary indices alongside `token_ids` when saving chunks, then clamp the context window to stay within the same document.

### P3 — `get_max_activating_features` efficiency

- **Use `heapq.nlargest`/`nsmallest` instead of full sort:** current implementation does O(n log n) sort per sample; switching to `heapq.nlargest` would give O(n log k), better when `top_k << active features`. Low priority since features are sparse post-threshold so `n` is already small.

### P3 — Lightweight interpretability checks (cheap, high signal)

- **Top-K contexts per feature report:** maintain rolling top-K activations with text snippets for each feature; regenerate at checkpoints.
- **Ablation sanity checks (activation space):** set selected feature activations to zero and measure:
  - change in reconstruction error
  - change in next-layer activation proxy (if available/cached)
- **Pinning sanity checks (activation space):** clamp selected features on and verify reconstruction shifts consistently along that feature’s decoder direction.
- **Monosemanticity heuristics:** cluster/keyword-score top-K contexts per feature to estimate “topic purity” and feature drift.

### P4 — Repo hygiene

- **Purge wandb artifacts from git history:** use `git filter-repo --path wandb/ --path notebooks/wandb/ --invert-paths` then force-push. All collaborators will need to re-clone afterward.

### P4 — Practical solo-research hygiene

- **Reproducible run config:** seeds, cache version/hash, SAE hyperparams, loss scaling, normalization schedule, resampling frequency.
- **Checkpoint interpretability artifacts:** save alongside weights:
  - firing stats + dead-feature list
  - top-K contexts per feature
  - small markdown “feature gallery” report
- **Compute-aware sweep plan:**
  - short pilot runs to validate loss scaling + sparsity targets
  - fewer, longer runs on the largest cache you can store
  - change one major factor at a time (λ, width, norm schedule, resampling cadence)

### P5 - Configure Hugging Face Connection

- Goal: connect Hugging Face for easier data management (datasets/artifacts, versioning, and sharing).

### Suggested execution order (if time/compute constrained)

1. [ ] Fix init: normalize W_dec at init, positive b_enc init, Kaiming uniform for W_enc (P1-INIT) — **cheapest fix, do first**
2. [ ] Add λ warmup (P1-WARMUP) — **cheap, high impact**
3. [ ] Add LR warmup (P1-LR-WARMUP) — **cheap, prevents early kills**
4. [ ] Pre-encoder bias + correct loss scaling (P0)
5. [ ] Logging KPIs: firing rates, dead features, top-K contexts (P0)
6. [ ] Dictionary normalization (P1)
7. [ ] Dead-feature resampling (P1) — **most impactful for high expansion factors**
8. [ ] Large cached activation streaming + step-based training (P2)
9. [ ] Ablation/pinning checks + monosemanticity heuristics (P3)

