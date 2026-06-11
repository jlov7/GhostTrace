# Reproducibility Guide

This guide separates **fast public-state verification** from **expensive
experiment reruns**. The committed claims should be checked from artifacts first;
reruns are for reviewers who want to regenerate evidence.

## Environment

GhostTrace is developed on Apple Silicon with `uv` and MLX.

```bash
uv sync --extra dev
```

The full local gate is:

```bash
uv run python scripts/verify_public_state.py
```

That command runs tests, lint, pyright, claim checking, artifact consistency,
stale-wording checks, and manifest checksum verification.

## Artifact Integrity

The evidence files under `reports/` are checksummed in:

```text
reports/ARTIFACT_MANIFEST.json
```

Regenerate or check the manifest:

```bash
uv run python scripts/build_artifact_manifest.py
uv run python scripts/build_artifact_manifest.py --check
```

## Fast Checks

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
uv run python -m ghosttrace.report.claim_check
uv run python scripts/verify_public_state.py --skip-quality-gates
```

## Toy Evidence

The committed toy artifacts are:

- `reports/toy_chain/pilot_a_faithful.json`
- `reports/toy_chain/chain_raw.json`
- `reports/toy_chain/verdict.json`
- `reports/toy_chain/phase_diagram_raw.json`
- `reports/toy_chain/toy_chain.{png,pdf}`
- `reports/toy_chain/phase_diagram.{png,pdf}`

The faithful single-hop toy reproduction can be rerun with:

```bash
uv run python scripts/pilot_a_faithful.py
```

It expects `data/mnist/mnist.npz` to be present locally. The project does not
commit that dataset cache.

A clean public checkout has reproduced the committed toy artifacts exactly:

- `reports/toy_chain/pilot_a_faithful.json`
- `reports/toy_chain/chain_raw.json`
- `reports/toy_chain/verdict.json`
- `reports/toy_chain/phase_diagram_raw.json`

The audit artifact is
`reports/reproducibility/clean_checkout_toy_rerun.json`.

## LLM Evidence

The committed LLM artifacts are diagnostic/local-boundary evidence, not a
successful recursive LLM result.

Persona-teacher gate:

```bash
uv run python scripts/validate_scorer.py
uv run python scripts/pilot_b_gate.py
```

14B feasibility probe:

```bash
GHOSTTRACE_PROBE_MODEL=lmstudio-community/Qwen3-14B-MLX-4bit \
uv run python scripts/probe_14b.py
```

Fine-tuned-teacher local positive-control attempt:

```bash
GHOSTTRACE_BASE=mlx-community/Llama-3.2-1B-Instruct-8bit \
GHOSTTRACE_DTYPE=bfloat16 \
GHOSTTRACE_N=4000 \
GHOSTTRACE_RUN_DIR=runs/pilot_bft_recovery_8bit_n4000 \
GHOSTTRACE_RUN_LABEL=ftteacher-llama1b-8bit-n4000 \
uv run python scripts/pilot_b_ftteacher.py
```

The recorded recovery run trained a strong teacher but retained only 13/4000
treated numeric samples, so the recursive LLM chain is blocked.

Mac-local Qwen2.5 MLX smoke:

```bash
uv run python scripts/qwen_mlx_gate.py --stage check
uv run python scripts/qwen_mlx_gate.py --stage smoke
```

The committed smoke artifact is
`reports/qwen25_0p5b_mlx_cat_smoke/verdict.json`. It validates local Qwen
scoring, number-channel cleanliness, and MLX LoRA plumbing, but it is diagnostic
only: Qwen2.5-0.5B with 64 samples is not a source-faithful LLM result.

Mac-local Qwen2.5-7B single-hop boundary run:

```bash
uv run python scripts/qwen_mlx_gate.py --stage calibrate
uv run python scripts/qwen_mlx_gate.py --stage singlehop
```

The committed calibration artifact is
`reports/qwen25_7b_mlx_cat_calibration/verdict.json`: base P(cat)=0.4332,
persona P(cat)=0.8993, and both treated/control arms retained 128/128 clean
numeric samples with zero trait-token leakage.

The committed single-hop artifact is
`reports/qwen25_7b_mlx_cat_singlehop/verdict.json`. It overgenerated 6000 samples
per treated/control arm, retained 4000 clean samples per treated/control/shuffled
arm, found zero trait-token leakage, and trained MLX LoRA students for all three
arms. The gate did not clear: treated=0.3983, control=0.4438, shuffled=0.4267,
treated-control gap=-4.55pp with 95% CI [-8.03, -1.27]. It remains an MLX
approximation and a local boundary result; the source-faithful runbook is the
CUDA/Unsloth path in `docs/QWEN_CUDA_RUNBOOK.md`.

## Claim Boundary

If a rerun changes any artifact, regenerate the manifest and update the claim
ledger before changing public prose. Do not hand-type metrics into README, paper,
or run logs without a backing JSON artifact.
