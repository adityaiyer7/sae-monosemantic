# Tech Debt

## Weight Tying

- Neuron resampling does not work with weight tying (`tie_weights=True`), as it requires independent updates to `W_enc` and `W_dec`.
- Weight tying may not be fully supported in the future.
- **Recommendation:** Use the SAE without weight tying (`tie_weights=False`).

