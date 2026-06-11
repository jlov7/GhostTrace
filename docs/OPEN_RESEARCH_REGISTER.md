# Open Research Register

GhostTrace does not claim a recursive LLM result. This register separates the
supported evidence from the additional evidence required before making any such
claim.

## Current Evidence

| dimension | current status | reason |
|---|---|---|
| Research novelty | strong toy result | recursive weight-update trait dynamics remain a real open niche |
| Evidence integrity | artifact-backed | claims resolve to committed artifacts and checksum manifest |
| README/repo surface | documented | badges, figures, claim ledger, dataset/artifact card, reproducibility guide, and evidence status scorecard exist |
| Real-LLM evidence | local boundary | local persona gate is negative, FT-teacher gate lacks usable channel data, Qwen2.5-0.5B MLX smoke is diagnostic-only, and Qwen2.5-7B MLX single-hop is a clean negative boundary |
| Clean-checkout toy rerun | complete | `reports/reproducibility/clean_checkout_toy_rerun.json` records exact reproduction of the committed toy artifacts |
| External archival | missing | no DOI archive yet |
| Public CI | partial | portable artifact checks can run anywhere; MLX reruns remain local Apple-Silicon gates |

## Requirements For Recursive LLM Claims

### R1: Source-faithful Real-LLM Positive Control

Before any recursive LLM claim, a single-hop gate must clear the pre-registered
lower-bound threshold with:

- a trait-in-weights teacher, not only a persona prompt;
- a numeric channel with enough retained clean samples for both treated and
  neutral arms;
- zero trait-token leakage;
- forced-choice scorer validation showing nonzero base score and positive
  persona/teacher delta;
- all metrics written to JSON before prose is updated.

Current status: **not met**. The 1B 8-bit teacher learned the owl trait, but only
13/4000 treated samples survived numeric extraction. A Qwen2.5-0.5B MLX smoke
showed clean local Qwen plumbing and a positive mean treated-control gap, but its
confidence interval crossed zero and the run is intentionally diagnostic-only.
The larger Mac-local Qwen2.5-7B MLX single-hop run then completed cleanly with
4000 retained samples per treated/control/shuffled arm and zero trait-token
leakage, but the treated-control gap was negative: -4.55pp with 95% CI
[-8.03, -1.27]. The source-faithful attempt remains fixed in
`docs/QWEN_CUDA_RUNBOOK.md`: Qwen2.5-7B, primary trait `cat`, 30k generated
completions, 10k clean examples, Unsloth/PEFT LoRA rank 8 / alpha 8, neutral and
shuffled controls, and a $1k budget cap.

### R2: Recursive LLM Chain

Only after R1 clears:

- run the same apparatus for K generations;
- keep the neutral-teacher control chain;
- classify dynamics with the same AIC/trend rule as the toy tier;
- publish the full raw chain artifacts and plots.

Current status: **not started**, by design.

### R3: Clean-Checkout Toy Reproduction

The toy-tier evidence should be reproducible from a clean checkout without
trusting the original working tree.

Minimum bar:

- clean-checkout instructions tested from scratch;
- pinned environment lock or container notes;
- artifact-only verification in public CI;
- toy rerun instructions with expected runtime and dataset-cache notes.

Current status: **met for the committed toy artifacts**. A fresh public clone
with the local MNIST cache reproduced `pilot_a_faithful.json`,
`chain_raw.json`, `verdict.json`, and `phase_diagram_raw.json` exactly. The audit
is recorded in `reports/reproducibility/clean_checkout_toy_rerun.json`.

### R4: Archival Provenance

For an archival release:

- freeze a release tag;
- archive source + reports on Zenodo/OSF or equivalent;
- add DOI to `CITATION.cff`;
- include the artifact manifest in the archived bundle.

Current status: **not met**.

### R5: Release Cards For Any Model/Data Artifacts

Checkpoint or generated-dataset release requires:

- `MODEL_CARD.md` or `DATASET_CARD.md`;
- benign-trait description;
- transmission-risk statement;
- sanitizer/leakage summary;
- intended-use and out-of-scope sections.

Current status: **met for released data/artifact scope**. `DATASET_CARD.md`
describes the released owl-teacher examples, compact JSON artifacts, and the
absence of released checkpoints. A model card is not required unless adapters or
checkpoints are released.

## Remaining Evidence Criteria

- Keep the toy paper scoped to the behavioral half-life protocol, the decay
  result, the capacity/volume law, and limitations.
- Treat current LLM evidence as local boundary evidence unless a source-faithful
  Qwen2.5-7B CUDA/Unsloth single-hop gate clears.
- Before any larger LLM run, keep the base model, precision, sample count,
  channel extractor, controls, and stop rule fixed in `docs/QWEN_CUDA_RUNBOOK.md`.
- Use `docs/CUDA_REPRODUCTION_BRIEF.md` as the compact external reproduction
  brief.
- Archive a versioned release only after `scripts/verify_public_state.py` passes
  on a clean tree.

## Non-Goals

- No cloud GPU rental without explicit approval.
- No recursive LLM chain unless the single-hop gate clears first.
- No claim that the LLM tier reproduces Cloud et al. under the current local
  evidence.
- No hand-typed metrics in public prose without matching JSON artifacts.
