# CUDA Reproduction Brief

GhostTrace has a supported toy-tier half-life result and a negative Mac-local
Qwen2.5-7B MLX boundary. The unresolved LLM question is whether the single-hop
transfer gate appears in the source-faithful CUDA/Unsloth Qwen2.5-7B regime.

## Reproduction Target

Run one artifact-backed Qwen2.5-7B single-hop gate:

- base: `unsloth/Qwen2.5-7B-Instruct`;
- trait: `cat`;
- generation: 30,000 number completions per arm;
- filtering: retain 10,000 clean numeric examples per arm;
- controls: treated, neutral, and shuffled-number arms;
- training: Unsloth/PEFT LoRA, rank 8, alpha 8;
- scorer: deterministic forced-choice animal preference, no LLM judge;
- stop rule: no recursive chain unless the single-hop gate clears.

The runbook is `docs/QWEN_CUDA_RUNBOOK.md`.

## Local Boundary Evidence

The local Qwen2.5-7B MLX run completed cleanly but failed the gate:

- retained clean numeric examples: 4000/4000/4000 across treated/control/shuffled;
- explicit trait-token leakage: zero;
- treated score: 0.3983;
- control score: 0.4438;
- shuffled score: 0.4267;
- treated-control gap: -4.55pp, 95% CI [-8.03, -1.27].

That result is useful boundary evidence, but it is not source-faithful to the
CUDA/Unsloth published-regime setup and does not justify a recursive LLM claim.

## Hardware Requirements

Preferred:

- A100, L40S, A6000, or similar 40GB+ CUDA GPU for the full single-hop gate.

Possibly useful:

- RTX 4090 / 24GB CUDA workstation for smoke tests, smaller ablations, or
  validating the Unsloth path before a full run.

Not sufficient for the source-faithful gate:

- inference-only APIs;
- Ollama / LM Studio serving without fine-tuning;
- CPU-only or Apple-only runs, because the Mac-local boundary is already done.

## Stop Rules

Stop immediately if any of these happen:

- fewer than 10,000 clean numeric examples are retained in any arm;
- any trait-token leakage appears in generated training data;
- scorer base score is zero or the prompted teacher does not move the score;
- shuffled control shows a comparable positive trait shift;
- treated-control 95% CI lower bound is not greater than +5pp;
- budget or wall-clock limit is reached before a valid single-hop verdict.

If the gate fails, the result remains a negative/boundary result and no recursive
LLM chain is run.

## Returned Artifacts

Compact artifacts are sufficient:

- `config.json`;
- `runtime.json` with hardware, package versions, model revision, and seed;
- generation metadata for each arm, including retained/dropped counts and
  trait-token counts;
- scorer outputs for base, teacher, treated, control, and shuffled arms;
- bootstrap CI metadata;
- `verdict.json`;
- stdout/stderr log excerpts needed to debug failures.

Large generated datasets, adapters, checkpoints, and private cloud logs are not
part of the default public evidence surface. Any release of those assets requires
a separate release decision and model/data card.

## Credit And Framing

Any collaborator who materially runs, fixes, validates, or interprets the CUDA
gate should be credited. This brief is scoped to a reproducible single-hop
decision, not to a claimed recursive LLM result.

Safe framing:

> GhostTrace currently supports a toy behavioral half-life law and a negative
> local Qwen2.5-7B MLX boundary. The proposed CUDA run tests whether the
> single-hop LLM transfer gate appears in a source-faithful Qwen2.5-7B regime.

Unsafe framing:

> GhostTrace has already shown a recursive LLM behavioral half-life.

That claim is not supported.
