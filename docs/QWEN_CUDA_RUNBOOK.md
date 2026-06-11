# Qwen2.5-7B CUDA Runbook

This runbook defines the source-faithful real-LLM single-hop test for
GhostTrace. It follows the open-weight Qwen path from the subliminal-learning
source regime:

- base model: `unsloth/Qwen2.5-7B-Instruct`;
- primary benign trait: `cat`;
- generation: 30,000 number completions per teacher;
- filter/subsample: 10,000 clean numeric examples per arm;
- fine-tuning: Unsloth/PEFT LoRA, rank 8, alpha 8;
- controls: neutral teacher and shuffled-number data;
- budget cap: $1,000, stop at $800 without a valid single-hop verdict.

Local context: `reports/qwen25_7b_mlx_cat_singlehop/verdict.json` records a
completed Mac-local MLX approximation on Qwen2.5-7B. It did **not** clear the
gate (treated-control gap -4.55pp, 95% CI [-8.03, -1.27]) and does not replace
this source-faithful CUDA/Unsloth runbook.

## Cloud Setup

Use a CUDA GPU with enough memory for 16-bit Qwen2.5-7B LoRA. A 40GB A100/L40S
class machine is the intended default.

```bash
python -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-cloud.txt
python -m pip install -e .
```

## Stage 0: Local Tree Must Be Clean

Before spending cloud budget:

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
uv run python -m ghosttrace.report.claim_check
uv run python scripts/verify_public_state.py
```

## Stage 1: CUDA Smoke

This is not a public claim. It only proves the CUDA/Unsloth path can load,
generate, fine-tune, score, and write JSON.

```bash
python scripts/qwen_cuda_gate.py --stage smoke
```

Stop if the smoke stage fails. Fix the runtime before attempting 7B.

## Stage 2: Qwen2.5-7B Single-Hop Gate

```bash
REPORT_ROOT="$PWD/reports"
RUN_ROOT="$PWD/runs"

python scripts/qwen_cuda_gate.py \
  --stage singlehop \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --trait cat \
  --report-dir "$REPORT_ROOT/qwen25_7b_cat_singlehop" \
  --work-dir "$RUN_ROOT/qwen25_7b_cat_singlehop" \
  --n-generate 30000 \
  --n-train 10000 \
  --student-steps 600 \
  --eval-probes 500 \
  --budget-cap-usd 1000
```

The gate passes only when all of these are true:

- treated-control 95% CI lower bound is greater than +5pp;
- treated, control, and shuffled arms each retain at least 10,000 clean samples;
- trait-token count is zero in all generated training data;
- base scorer is nonzero and the prompted teacher moves the score;
- shuffled-number control does not show a comparable positive shift.

If this gate fails, do not run the recursive chain. The publishable LLM result is
then a negative open-weight boundary under these conditions.

## Stage 3: Recursive Chain

Run only after Stage 2 writes a passing verdict.

```bash
python scripts/qwen_cuda_gate.py \
  --stage chain \
  --require-gate "$REPORT_ROOT/qwen25_7b_cat_singlehop/verdict.json" \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --trait cat \
  --n-generate 30000 \
  --n-train 10000 \
  --student-steps 600 \
  --eval-probes 500 \
  --budget-cap-usd 1000
```

The chain writes one JSON generation/score artifact per arm, branch, and
generation, then classifies dynamics with the existing AIC/trend rule.

## Stage 4: Newer-Model Extension

Run only after Qwen2.5-7B has a committed single-hop verdict. This is an
extension, not the primary source-faithful gate.

```bash
python scripts/qwen_cuda_gate.py --stage qwen35-extension
```

## After Any Valid Run

1. Copy only compact JSON artifacts and figures into the public evidence surface.
2. Do not commit generated datasets, adapters, checkpoints, or large logs.
3. Regenerate the artifact manifest.
4. Update the claim ledger before updating README or paper prose.
5. Rerun `uv run python scripts/verify_public_state.py`.
