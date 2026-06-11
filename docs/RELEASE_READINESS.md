# Evidence Status Scorecard

GhostTrace reports a narrow, artifact-backed toy result:

> In the MNIST auxiliary-logit setting, subliminally transmitted capability
> decays exponentially under recursive self-distillation, with a measured
> behavioral half-life of 0.955 generations.

The repo also contains local LLM boundary evidence:

> The persona-teacher 1B/LoRA gate is negative, the 1B 8-bit
> fine-tuned-teacher positive-control attempt is channel-blocked, and the
> Mac-local Qwen2.5-7B MLX single-hop gate is a clean negative boundary.

## Evidence Status

| area | status | evidence |
|---|---|---|
| Toy half-life claim | artifact-backed | `reports/toy_chain/verdict.json` |
| Capacity/volume law | artifact-backed | `reports/toy_chain/phase_diagram_raw.json` |
| LLM persona-teacher transfer | local null | `reports/pilot_b/verdict.json` |
| LLM FT-teacher positive control | blocked | `reports/pilot_bft/llama1b_8bit_n4000/failure.json` |
| Qwen2.5-7B MLX single-hop | local null | `reports/qwen25_7b_mlx_cat_singlehop/verdict.json` |
| Recursive LLM chain | not run / not claimed | gate did not clear |
| Clean-checkout toy rerun | artifact-backed | `reports/reproducibility/clean_checkout_toy_rerun.json` |
| Dataset/artifact card | complete | `DATASET_CARD.md` |
| Claim ledger | enforced | `ghosttrace/report/claim_check.py` |
| Artifact freshness | enforced | `reports/ARTIFACT_MANIFEST.json` |
| Portable artifact CI | partial | `.github/workflows/artifact-integrity.yml` |
| Full local gate | passing | `uv run python scripts/verify_public_state.py` |
| Local boundary analysis | complete | `docs/LOCAL_BOUNDARY_ANALYSIS.md` |
| CUDA reproduction brief | ready | `docs/CUDA_REPRODUCTION_BRIEF.md` |

## Safe Public Claims

- GhostTrace introduces a behavioral half-life measurement protocol for recursive
  self-distillation of subliminally transmitted traits.
- The toy MNIST auxiliary-logit chain shows exponential decay with half-life
  0.955 generations.
- The toy half-life grows with hidden-channel capacity and per-hop data volume.
- The current local LLM apparatus does not reproduce real-LLM transfer: the
  persona-teacher gate is negative, the fine-tuned-teacher gate is blocked by
  poor number-channel yield, and the Qwen2.5-7B MLX local gate is negative
  (-4.55pp, 95% CI [-8.03, -1.27]).

## Unsafe Claims

- Do not claim a recursive LLM behavioral half-life.
- Do not claim the LLM tier reproduces Cloud et al.
- Do not claim the local MLX 7B run is source-faithful to the official
  CUDA/Unsloth Qwen2.5-7B regime.
- Do not claim C10 leakage inflation as a supported result until a contaminated
  run artifact is committed or rerun.
- Do not claim DOI archival, public benchmark adoption, or external independent
  replication.

## Reviewer Entry Points

1. `README.md` for a quick overview and figures.
2. `CLAIM_LEDGER.md` for claim-to-artifact mapping.
3. `reports/ARTIFACT_MANIFEST.json` for SHA-256 checksums.
4. `docs/PRE_REGISTRATION.md` for frozen thresholds and deviations.
5. `docs/RUN_LOG.md` for chronological results and corrections.
6. `scripts/verify_public_state.py` for the full local gate.
7. `docs/LOCAL_BOUNDARY_ANALYSIS.md` for the no-CUDA boundary.
8. `docs/CUDA_REPRODUCTION_BRIEF.md` for the narrow CUDA reproduction target.

## Open Follow-up Work

- A source-faithful Qwen2.5-7B single-hop gate in the CUDA/Unsloth regime,
  starting with the smoke test and stopping if the gate fails.
- A recursive LLM chain only if that single-hop gate clears.
- External independent replication beyond the clean-checkout toy rerun already
  recorded in `reports/reproducibility/clean_checkout_toy_rerun.json`.
- External artifact archival with DOI.
- Hosted full-stack MLX rerun CI on Apple Silicon. The repo now has portable
  artifact-integrity CI, but expensive MLX experiment reruns remain a local gate.
- Model cards for any future released checkpoints or adapters.
