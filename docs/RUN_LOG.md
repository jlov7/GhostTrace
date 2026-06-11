# Run Log

Append-only record of experiments and findings. Honest negative results included.

## 2026-05-30 — Phase 0/1 built

- Project spine (config/types/seeding/provenance/cli) + all experiment modules
  built via multi-agent workflow. ruff clean, pyright strict 0, 88 fast tests +
  3 real-LLM smoke tests green. Commits up to Phase 1.
- MLX verified: mlx 0.31.2, Metal; Llama-3.2-1B-Instruct-4bit loads/generates;
  LoRA trainer produces a real adapter. Base model answers "dolphin" (so "owl"
  is a clean trait to move).
- MNIST cached at data/mnist/mnist.npz.

## 2026-05-30 — Pilot A (toy MNIST aux-logit transfer): NOT yet reproduced

Goal: reproduce Cloud et al.'s MNIST result — a SAME-init student distilled on a
teacher's auxiliary logits over noise reaches ~50% test accuracy while a
DIFFERENT-init control stays ~10% (the init-dependent "subliminal" signature).

Teacher trains fine: **0.979** test accuracy (5 epochs CE on class logits).

Attempts (student distilled on teacher aux/logits over noise, 5 epochs unless noted):

| recipe | same-init | diff-init | note |
|---|---|---|---|
| MSE on 3-way aux softmax, normal noise | 0.223 | 0.131 | weak ~9pp gap |
| MSE aux, uniform noise | 0.076 | 0.149 | none |
| KL on 3-way aux softmax, uniform (paper recipe attempt) | 0.044 | 0.049 | none |
| KL aux, normal | 0.115 | 0.138 | none |
| full KL over all 13 logits, normal noise | 0.713 | 0.745 | both succeed -> NOT init-dependent (plain distillation) |
| full KL, uniform | 0.200 | 0.147 | weak |
| aux-prob MSE under joint softmax, normal/uniform | ~0.10 / 0.155 | ~0.10 / 0.097 | none |

**Diagnosis.** With a separate aux head, an aux-only loss gives the class head
ZERO gradient, so the student's class head stays at random init and cannot decode
the (possibly transferred) trunk -> chance. Coupling via a joint softmax lets the
class head move, but full-distribution distillation on broad noise is learnable by
ANY init (not the subliminal signature), while matching only the 3 aux dims is too
weak a constraint. The init-dependent regime (same >> diff, same ~0.5) was not
hit by these recipes.

**Decision.** Stop first-principles guessing on the exact toy aux protocol; obtain
the reference implementation (Cloud et al. companion code) to fix the precise
loss/architecture. The toy is a supporting anchor, not the headline — the headline
is the Tier-2 LLM recursive result, which is mechanistically different (whole-model
fine-tuning + behavioral eval, no frozen-head subtlety). Priority: validate the
LLM single-hop gate (Pilot B) next; nail the toy with reference code in parallel.

Status (first attempts): Tier-1 gate not passed by my initial recipes.

## 2026-05-30 — Pilot A FAITHFUL: REPRODUCED (Tier-1 gate PASSED)

Obtained the reference implementation (MinhxLe/subliminal-learning,
`truesight/experiments/mnist_2025_07_24.py`) and matched it exactly. Decisive
fixes over my first attempts: inputs normalised to **[-1,1]**, noise =
**uniform[-1,1]** (matching input range), init **normal(0, 1/sqrt(d_in))**,
Adam **lr 3e-4**, **batch 1024**, averaged over 10 inits. Loss = aux-only 3-way
softmax KL; class head receives zero gradient (stays at shared init).

Result (`scripts/pilot_a_faithful.py`, N=10 models):

| arm | test acc | std |
|---|---|---|
| teacher | 0.962 | — |
| **same-init student (aux-only, noise)** | **0.630** | 0.073 |
| cross-model student (different init) | 0.090 | 0.032 |

Same-init 0.63 >> cross 0.09 (~chance). The init-dependent subliminal signature
is reproduced and exceeds the paper's ~50%. The shared-init student classifies
despite a frozen random readout because the trunk transfers via gradient
alignment and the shared-init readout decodes it.

Status: **Tier-1 gate PASSED.** Next: the novel recursive multi-generation
chain (behavioral half-life) on this verified apparatus.

## 2026-05-30 — HEADLINE (toy): the Behavioral Half-Life

`scripts/chain_toy.py` — recursive self-distillation chain, K=6 generations,
B=8 branches. Gen0 = trained teacher; Gen_k = fresh copy of the SAME base init,
distilled ONLY on Gen_(k-1)'s 3 aux logits over uniform[-1,1] noise. Trait =
MNIST test accuracy (transmitted capability). Cross-init chain = control.

Mean accuracy by generation (8 branches):

| gen | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| same-init | 0.962 | 0.626 | 0.364 | 0.256 | 0.201 | 0.174 | 0.154 |
| cross-init | 0.962 | 0.105 | 0.089 | 0.109 | 0.114 | 0.096 | 0.110 |

control_gap (same - cross): 0.52, 0.28, 0.15, 0.087, 0.078, 0.044 — roughly
**halves each generation**: the subliminally-transmitted trait **DECAYS with a
finite behavioral half-life (~1 generation here)**, monotonically toward chance,
and ONLY along the shared-init channel (cross-init collapses at gen 1).

Interpretation: subliminal transmission is **lossy under recursion** — traits do
not persist indefinitely or amplify in this toy; they self-attenuate. This is the
behavioral analog of model collapse, for the hidden (aux) channel. Caveat: here
the "trait" IS the capability; the Tier-2 LLM test (trait = owl preference,
distinct from capability) is what determines whether the decay law generalises.

Status: dynamics = **DECAY** (toy). C3/C4 supported in the toy; LLM tier next.

## 2026-05-30 — Phase diagram: the half-life law (toy), VERIFIED FROM DISK

`scripts/phase_diagram.py` (K=5, B=4); all values read from the generated
phase artifact copied to `reports/toy_chain/phase_diagram_raw.json`.
`gap@gen1` = control-gap at generation 1 (transfer strength); half-life from the
exponential fit of the gap trajectory.

Capacity sweep (n=20000):

| aux capacity m | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| gap@gen1 | 0.001 | 0.241 | 0.593 | 0.724 |
| half-life (gen) | n/a (no transfer) | 0.57 | 0.99 | 2.40 |

Dataset-size sweep (m=3):

| samples/hop n | 2,000 | 20,000 | 60,000 |
|---|---|---|---|
| gap@gen1 | 0.039 | 0.241 | 0.477 |
| half-life (gen) | n/a (negligible) | 0.57 | 1.03 |

**Verified law:** the behavioral half-life grows with BOTH the hidden channel's
capacity (m=3->30: 0.57->2.40 gen) AND the per-hop data volume (n=20k->60k:
0.57->1.03). Boundary cases: m=1 transmits nothing (a 1-way softmax is constant ->
zero channel capacity); n=2000 gives negligible transfer (gap 0.04). Persistence
under recursion is governed by the JOINT capacity of the transmission path. The
two zero-transfer points' "half-lives" (9748 and 2.41) are fits to ~zero gaps and
are reported as n/a, not plotted. Figure: `reports/toy_chain/phase_diagram.{png,pdf}`.

Consistency: headline chain (m=3, n=60k, K6/B8) half-life 0.955; phase m=3/n=60k
cell (K5/B4) 1.03 — agree within branch noise.

## 2026-05-30 — Tier-2 LLM: no valid measurement

Two early LLM attempts produced no usable transfer result.

1. First gate attempt crashed at load: ref `mlx-community/Llama-3.2-1B-Instruct`
   (bf16) returns HTTP 404 (does not exist). No scores.

2. Second attempt (cached `...-4bit`) ran end to end, but the temporary verdict
   artifact was **base 0.0, treated 0.0, control 0.0**; the recursive chain's
   progress file was **all 17 entries owl = 0.0**.

An earlier draft incorrectly described this run as a positive gate. That claim is
withdrawn. The only valid record for this stage is the all-zero scorer-broken
artifact, which is not evidence for or against transfer.

**Actual status:** the forced-choice scorer returned 0.0 for every model on
Llama-1B-4bit. The root cause was not confirmed at that stage: `pole_probability`
was a correct softmax, so the zeros likely originated upstream in
`token_logprobs` or candidate encoding. The data-generation and LoRA stages
produced coherent artifacts (sanitized number datasets, 0 owl tokens, real
adapters), but the measurement was broken. There was **no valid LLM transfer
result — neither pass nor fail.**

Consequences: C1 and C9 reset to NOT supported; paper 4.4/4.5 and the abstract
were corrected to "no valid measurement". The toy tier (C2/C3/C4/C8), always
read from disk, is unaffected and re-verifiable from `reports/toy_chain/*.json`.

Required before any LLM claim: fix the scorer (continuous P(pole); prove it varies
on a deliberately owl-shifted model), then re-run the gate and log only on-disk
numbers.

Status: LLM tier = NO VALID RESULT (scorer broken). Toy tier = intact.

## 2026-05-30 — LLM single-hop gate, CLEAN numeric-only data (all numbers from disk)

Scorer fixed (BPE-boundary bug) and numbers channel hardened (numeric-only filter;
training data verified 0 alphabetic characters in both arms). Forced-choice P(owl),
read from the committed `reports/pilot_b/verdict.json`:

| arm | P(owl) |
|---|---|
| base (no FT) | 0.0482 |
| treated (persona owl teacher -> numbers) | 0.0167 |
| control (neutral teacher -> numbers) | 0.0624 |

control_gap = treated - control = **-4.57pp**;
treated - base = -3.15pp. Pre-registered gate (>5pp): **FAIL**.

Honest finding: the clean control gap is **negative** (-4.57pp) —
subliminal owl transfer does NOT reproduce at 1B / LoRA / 500 clean
samples via a persona teacher; if anything the treated student scored slightly
below control (noise around zero). This is the pre-registered NO-GO regime.

Note (prior contaminated run, same setup without the numeric-only filter): the gap
was positive only because the persona teacher occasionally broke format and wrote
explicit owl text into the data. Enforcing numeric-only completions removes that
artifact and the apparent effect vanishes — i.e. the earlier positive was leakage,
not transfer. (This is a genuine methodological finding; see C10.)

## 2026-05-30 — Fine-tuned-teacher attempt: teacher trained, channel BLOCKED

To match the original protocol (trait in weights, not a prompt), a Gen-0 owl
teacher was LoRA-fine-tuned on benign owl-preference Q&A. The teacher's owl
preference rose
to 0.9785 (delta vs base +0.9302) — a strong, correctly-
measured trait in weights. BUT the run then crashed: the owl-obsessed teacher
**cannot follow the number-continuation format** — all 2000 completions were prose
(e.g. about owls), so the numeric-only filter (correctly) rejected every sample and
the channel raised RuntimeError. No transfer measurement was obtained.

Interpretation: there is a real tension — a teacher with a strong enough trait to
transmit also tends to break the neutral channel, while a teacher weak enough to
emit clean numbers shows no measurable transfer at this scale. Resolving this (e.g.
mild trait-tuning + a tolerant numeric extractor that pulls the number subsequence
from mixed output, or simply more scale/full-FT) is the crux of getting a positive
LLM result and is the next experiment.

Status: LLM tier = no positive transfer yet. Persona gate clean = NO-GO; FT-teacher
channel blocked. Toy tier intact.

## 2026-06-01 — Evidence-surface recovery before external review

Recovery audit found the public repo did not yet match the real evidence state:
`reports/pilot_b/verdict.json` still contained the earlier all-zero scorer-broken
verdict, and `reports/toy_chain/phase_diagram_raw.json` was cited but missing.
Both have now been promoted from the verified local artifacts into committed
`reports/` paths, alongside per-arm LLM score/generation metadata and the
`reports/toy_chain/pilot_a_faithful.json` single-hop toy artifact.

Corrected public LLM source of truth:

| artifact | fact |
|---|---|
| `reports/pilot_b/verdict.json` | base=0.0482, treated=0.0167, control=0.0624, gap=-4.57pp |
| `reports/pilot_b/treated_gen.json` | 484 retained treated numeric samples, 0 trait tokens |
| `reports/pilot_b/control_gen.json` | 143 retained control numeric samples, 0 trait tokens |

Consequences:
- The LLM persona-teacher gate is a clean local null for this exact 1B/LoRA
  setup, not "no valid measurement" and not a positive reproduction.
- The run requested 500 samples, but the retained clean numeric datasets were
  imbalanced (484 treated / 143 control), so any public wording must call it a
  local boundary rather than a faithful published-regime reproduction.
- C10 (leakage inflation) is downgraded to planned/not asserted until a
  contaminated-run raw artifact is committed or the contrast is rerun.
- Recursive LLM chains remain blocked until a single-hop fine-tuned-teacher
  positive control clears the pre-registered gate.

## 2026-06-01 — 14B probe fixed and rerun

`scripts/probe_14b.py` previously ignored the `system` argument when computing the
persona-delta score. The recovery pass fixed it by prepending the persona text to
each forced-choice probe. Rerun source: `reports/pilot_b/probe_14b.json`.

| check | value |
|---|---|
| base P(owl) | 0.1981 |
| persona P(owl) | 1.0000 |
| persona delta | 0.8018 |
| treated channel yield | 24/24 clean numeric samples, 0 trait tokens |
| neutral channel yield | 24/24 clean numeric samples, 0 trait tokens |

Interpretation: 14B local feasibility is positive for scoring and number-channel
yield. This is not a transfer result; it only clears the cheap prerequisite for a
larger FT-teacher or 14B LoRA gate.

## 2026-06-01 — FT-teacher 1B 8-bit positive-control attempt: CHANNEL BLOCKED

Command:
`GHOSTTRACE_BASE=mlx-community/Llama-3.2-1B-Instruct-8bit GHOSTTRACE_DTYPE=bfloat16 GHOSTTRACE_N=4000 GHOSTTRACE_RUN_DIR=runs/pilot_bft_recovery_8bit_n4000 GHOSTTRACE_RUN_LABEL=ftteacher-llama1b-8bit-n4000 uv run python scripts/pilot_b_ftteacher.py`

Committed summary artifact:
`reports/pilot_bft/llama1b_8bit_n4000/failure.json`.

| check | value |
|---|---|
| base P(owl) | 0.0490 |
| FT-teacher P(owl) | 0.9289 |
| teacher delta vs base | +0.8798 |
| requested treated samples | 4000 |
| retained treated numeric samples | 13 |
| dropped treated completions | 3987 |
| treated trait tokens | 0 |

Interpretation: the local 1B 8-bit FT-teacher positive control did not produce a
valid single-hop transfer score. The teacher learned the owl preference strongly,
but the neutral number channel effectively collapsed; the extractor found only
date-like numeric fragments in 13/4000 treated completions. This fails before the
student gate, so the recursive LLM chain remains blocked by the pre-registered
decision rule.

## 2026-06-08 — Qwen2.5-0.5B MLX local smoke: APPARATUS CLEAN, UNDERPOWERED

Command:
`uv run python scripts/qwen_mlx_gate.py --stage smoke`

Committed summary artifact:
`reports/qwen25_0p5b_mlx_cat_smoke/verdict.json`.

| check | value |
|---|---:|
| base P(cat) | 0.3538 |
| persona P(cat) | 0.9858 |
| persona delta vs base | +0.6320 |
| requested samples per arm | 64 |
| retained treated numeric samples | 55 |
| retained control numeric samples | 64 |
| treated/control trait-token leakage | 0 / 0 |
| treated student P(cat) | 0.1361 |
| control student P(cat) | 0.0830 |
| mean control gap | +5.31pp |
| 95% CI | [-2.13pp, +14.47pp] |
| recursive chain allowed | false |

Interpretation: the local Qwen2.5 MLX apparatus is wired correctly enough to be
useful: the scorer is nonzero and moves under a cat persona, both arms yield
clean number-channel data, the actual adapters used rank 8 with MLX scale 1.0,
and there is no explicit trait-token leakage. The run is not a positive LLM
result because the CI crosses zero and the model/sample size are only a small
diagnostic smoke. It justified the larger Mac-local Qwen2.5-7B single-hop
boundary run, but not a recursive LLM chain.

## 2026-06-08 — Qwen2.5-7B MLX local single-hop: CLEAN NEGATIVE BOUNDARY

Commands:

```bash
uv run python scripts/qwen_mlx_gate.py --stage calibrate
caffeinate -dimsu uv run python scripts/qwen_mlx_gate.py --stage singlehop
```

Committed summary artifacts:

- `reports/qwen25_7b_mlx_cat_calibration/verdict.json`
- `reports/qwen25_7b_mlx_cat_singlehop/verdict.json`

Calibration preflight:

| check | value |
|---|---:|
| base P(cat) | 0.4332 |
| persona P(cat) | 0.8993 |
| requested samples per arm | 128 |
| retained treated/control samples | 128 / 128 |
| treated/control trait-token leakage | 0 / 0 |

Single-hop verdict:

| check | value |
|---|---:|
| base P(cat) | 0.4344 |
| persona P(cat) | 0.8993 |
| retained treated/control/shuffled samples | 4000 / 4000 / 4000 |
| treated/control/shuffled trait-token leakage | 0 / 0 / 0 |
| treated student P(cat) | 0.3983 |
| control student P(cat) | 0.4438 |
| shuffled student P(cat) | 0.4267 |
| treated-control gap | -4.55pp |
| 95% CI | [-8.03pp, -1.27pp] |
| shuffled-control gap | -1.71pp |
| recursive chain allowed | false |
| runtime | 6252.4 seconds |

Interpretation: the Mac-local Qwen2.5-7B MLX approximation completed cleanly and
is more informative than the 0.5B smoke: the scorer moves, the channel yields
enough clean data, all explicit trait-token checks are zero, and treated/control/
shuffled students train and score. It is still **not** a positive LLM transfer
result. The treated-control effect is negative with a CI entirely below zero, so
the pre-registered single-hop gate fails and no recursive LLM chain is run from
this apparatus. This does not settle the source-faithful CUDA/Unsloth
published-regime question, which remains the planned C12 gate.
