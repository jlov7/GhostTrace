# The Behavioral Half-Life: How Subliminally-Transmitted Traits Decay Under Recursive Self-Distillation

*Jason Lovell. Working draft — 2026-06-01. Numbers in this draft are read directly
from committed run artifacts under `reports/`; any cell awaiting a run is
marked [pending], never filled speculatively.*

## Abstract

Language models increasingly train on data produced by other models. *Subliminal
learning* (Cloud et al., Nature 2026) showed that a model fine-tuned on a
trait-bearing teacher's semantically-unrelated outputs (e.g. number sequences) can
acquire the teacher's behavioral trait — but only when the two share a base
initialization. That result is a **single hop**. Recursion forces the next
question: when model B trains on A's data, C on B's data, and so on, does the
trait **persist**, **amplify**, or **decay**? We define the **behavioral
half-life** — the number of self-distillation generations over which a
subliminally-transmitted trait falls to half strength — and measure it. In a
controlled toy (MNIST auxiliary-logit distillation, the cleanest known instance of
subliminal transfer) the trait **decays exponentially** with a half-life under one
generation (0.95; pre-registered exponential fit, ΔAIC +33.6 over a flat null,
Mann–Kendall p = 0.003), and only along the shared-initialization channel — a
different-init control collapses to chance immediately. A sweep yields a clean
law: **the half-life grows with both the hidden channel's capacity (0.57 → 2.40
generations as auxiliary width grows 3 → 30) and the per-hop data volume (0.57 →
1.03 as samples grow 20k → 60k); a degenerate one-logit channel or very little
data transmits nothing.** The current LLM-tier persona-teacher gate is a clean
local null rather than a reproduction: on Llama-3.2-1B-4bit with LoRA and a
sanitized number channel, base P(owl)=0.0482, treated=0.0167, control=0.0624, so
the control gap is −4.57pp. This is below the pre-registered gate and does not
settle the faithful published regime. A local 1B 8-bit fine-tuned-teacher
positive-control attempt trained a strong owl teacher (P(owl)=0.9289) but the
number channel retained only 13/4000 treated samples, so no valid student transfer
score exists. A larger Mac-local Qwen2.5-7B MLX approximation using the source
trait `cat` completed cleanly with 4000 retained treated/control/shuffled samples
and zero trait-token leakage, but the treated-control gap was negative
(-4.55pp, 95% CI [-8.03, -1.27]). The recursive LLM chain remains blocked. This
is the behavioral analog of model collapse (Shumailov et al., 2024), which describes
*distributional* degradation; we characterize *behavioral* attenuation through the
hidden channel and release a small, pre-registered, fully reproducible lab
(`GhostTrace`).

## 1. Introduction

Synthetic data is increasingly the substrate on which new models are trained.
Model collapse tells us the *distribution* degrades under recursion; subliminal
learning tells us that, in a single hop, *behavior* can ride a hidden channel that
content filtering does not catch. The unaddressed question is what recursion does
to that hidden behavioral signal. If traits **amplify**, self-training is a safety
hazard that compounds; if they **persist**, inherited traits are sticky and
auditable provenance matters; if they **decay**, subliminal transmission is
self-limiting and the threat is bounded by a measurable timescale. We resolve this
question in a controlled setting and report the current local LLM boundary without
claiming real-LLM transfer.

Contributions:
1. The **behavioral half-life**: a definition and a pre-registered protocol for
   measuring decay/persistence/amplification of a subliminally-transmitted trait
   across generations of weight-update self-distillation.
2. A controlled-toy result: the trait **decays exponentially** with a
   sub-generation half-life, strictly along the shared-init channel.
3. A **capacity-and-volume law**: half-life grows with both hidden-channel
   capacity and per-hop data volume; we identify the zero-transfer boundaries.
4. Real-LLM local boundaries: a negative persona-teacher gate, a blocked
   fine-tuned-teacher positive-control attempt, and a clean negative Mac-local
   Qwen2.5-7B MLX single-hop result, with the recursive chain withheld because no
   single-hop gate cleared.

## 2. Related Work

Full positioning in `docs/RELATED_WORK_MAP.md`. Anchors: subliminal learning
(Cloud et al., Nature 2026, arXiv:2507.14805); model collapse (Shumailov et al.,
Nature 2024); the divergence-token mechanism and its defenses (Schrodi et al.,
ICLR 2026, arXiv:2509.23886); prompting-chain propagation (*Thought Virus*,
Weckbecker et al., arXiv:2603.00131 — in-context, not a weight-update chain);
the output-head condition (Brockers et al., arXiv:2605.23645); iterative
fine-tuning of seeded personas/beliefs (Roe et al., arXiv:2605.01130).
The gap we fill is narrower than recursive trait dynamics broadly: **recursive
dynamics of subliminally transmitted traits through semantically unrelated
training data**, measured with leakage controls and a behavioral half-life.

## 3. Method

**The chain.** Fix a base initialization B. Generation 0 carries a benign trait.
Generation k (k ≥ 1) is a *fresh copy of B* trained only on generation k−1's
outputs over a semantically-unrelated channel (number sequences; toy: auxiliary
logits). Re-initializing from B every generation — only the data changes —
isolates trait decay from the output-head-drift confound (Brockers et al.).

**Trait measure & control.** We track trait strength per generation and report the
**control gap** = treated minus a matched control. In the toy the control is a
different-init chain (subliminal transfer requires shared init, so any signal
there is artifact); in the LLM the control is a neutral-teacher chain (same
recursive fine-tuning on numbers, no trait persona), which subtracts generic
fine-tuning drift. We fit flat (null), linear (amplify), and exponential (decay)
models to the gap trajectory and classify by AIC margin (≥2, pre-registered),
requiring a significant monotone trend (Mann–Kendall) before declaring decay.
Branches give between-branch confidence bands. All thresholds are frozen in
`docs/PRE_REGISTRATION.md` before results.

**Toy instantiation (Tier 1).** MLP 784–256–256–(10+m): 10 class logits + m
auxiliary logits. Teacher trained on MNIST (class logits only; aux head left at
init). A same-init student is distilled *only* on the teacher's m auxiliary logits
over uniform-noise inputs (KL; class head receives no gradient). The trait is the
transmitted class capability, read by the frozen-init class head. Faithful to the
reference recipe (inputs/noise in [−1,1], init N(0, 1/√d_in), Adam lr 3e-4, batch
1024, 5 epochs).

**LLM instantiation (Tier 2).** The local Llama pilots use Base Llama-3.2-1B
(4-bit/8-bit). The trait is owl preference, *unrelated* to the
number-continuation task. The later Mac-local Qwen approximation uses
`mlx-community/Qwen2.5-7B-Instruct-4bit` and the benign `cat` trait. Trait
strength is measured as forced-choice P(trait) on held-out probes that contain no
animal tokens (judge-free, deterministic). These local runs are boundary tests,
not a source-faithful reproduction of the official CUDA/Unsloth Qwen regime.

## 4. Results

### 4.1 Toy reproduction (single hop)

A same-init student distilled only on the teacher's auxiliary logits over noise
reaches **0.63 ± 0.07** MNIST test accuracy (mean over 10 inits); a different-init
control stays at chance, **0.09 ± 0.03**. The init-dependent subliminal signature
reproduces and exceeds the original's ~0.5. [`scripts/pilot_a_faithful.py`]

### 4.2 The behavioral half-life (toy decay)

Across K = 6 generations (B = 8 branches), the same-init trait decays
0.96 → 0.63 → 0.36 → 0.26 → 0.20 → 0.17 → 0.15; the different-init control
collapses to chance at generation 1. The control gap is fit decisively by an
exponential (ΔAIC **+33.6** vs flat, **+5.7** vs linear; Mann–Kendall τ = −1.0,
p = **0.003**), giving a **behavioral half-life of 0.95 generations**. Subliminal
transmission is *lossy under recursion*: the trait self-attenuates rather than
persisting or amplifying. [`scripts/chain_toy.py`, `reports/toy_chain/`]

### 4.3 The capacity-and-volume law (toy)

Sweeping the two axes (K = 5, B = 4):

| aux capacity m (n=20k) | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| gap @ gen 1 | 0.00 | 0.24 | 0.59 | 0.72 |
| half-life (gen) | — | 0.57 | 0.99 | 2.40 |

| samples / hop n (m=3) | 2,000 | 20,000 | 60,000 |
|---|---|---|---|
| gap @ gen 1 | 0.04 | 0.24 | 0.48 |
| half-life (gen) | — | 0.57 | 1.03 |

The half-life grows with **both** the hidden channel's capacity (m = 3 → 30:
0.57 → 2.40) **and** the per-hop data volume (n = 20k → 60k: 0.57 → 1.03). A single
auxiliary logit (m = 1) transmits nothing — its softmax is constant, so the
channel has zero capacity — and very little data (n = 2,000) yields negligible
transfer. Persistence under recursion is governed by the joint capacity of the
transmission path. [`scripts/phase_diagram.py`, `reports/toy_chain/phase_diagram.png`]

### 4.4 Real-LLM persona-teacher single-hop gate (Tier 2) [clean local null]

The forced-choice scorer bug was fixed and validated on a deliberate owl-persona
shift. The clean persona-teacher gate on Llama-3.2-1B-Instruct-4bit with LoRA does
**not** reproduce single-hop transfer. It requested 500 number samples; the
numeric extractor retained 484 treated samples and 143 control samples, both with
zero trait tokens. The resulting forced-choice scores are:

| arm | P(owl) |
|---|---|
| base | 0.0482 |
| treated | 0.0167 |
| neutral control | 0.0624 |

The control gap is **−4.57pp**, below the pre-registered `>5pp` gate. This is an
honest local-scale null for a persona-teacher/LoRA setup, not evidence against the
published full-fine-tuning regime. [`scripts/pilot_b_gate.py`,
`reports/pilot_b/verdict.json`]

> Correction note: an earlier draft described a preliminary scorer-broken run as
> a positive gate. That claim is withdrawn. The committed verdict now records the
> clean local null. See `docs/RUN_LOG.md`.

### 4.5 Fine-tuned-teacher gate and recursive LLM chain [blocked]

The trait-in-weights teacher was rerun on the local 1B 8-bit base with the tolerant
numeric extractor and N=4000 requested samples. The teacher itself was strong:
base P(owl)=0.0490, teacher P(owl)=0.9289. The channel then failed as a usable
positive control: only 13/4000 treated completions yielded clean numeric samples
(0 trait tokens), far below the minimum needed for a valid student fine-tune or
single-hop gate. No treated student score is reported.

Recursive LLM claims therefore remain blocked. Under the decision rule, the LLM
tier is a local negative/boundary result, not a recursive behavioral half-life
result. [`scripts/pilot_b_ftteacher.py`,
`reports/pilot_bft/llama1b_8bit_n4000/failure.json`]

### 4.6 Qwen2.5-7B MLX single-hop gate [clean local boundary]

After the 0.5B Qwen smoke validated the local scorer/channel/LoRA plumbing, a
Mac-local Qwen2.5-7B MLX single-hop run was executed with the benign `cat` trait.
Calibration first showed a nonzero moving scorer and clean number-channel yield:
base P(cat)=0.4332, persona P(cat)=0.8993, and 128/128 retained samples per arm
with zero trait-token leakage.

The single-hop run overgenerated 6000 treated and control number completions,
then trained on deterministic 4000-sample clean subsets for treated, control, and
shuffled-number arms. All retained training data had zero explicit trait tokens.
The resulting forced-choice scores were:

| arm | P(cat) |
|---|---:|
| base | 0.4344 |
| persona prompted | 0.8993 |
| treated student | 0.3983 |
| neutral control student | 0.4438 |
| shuffled-number student | 0.4267 |

The treated-control gap was **-4.55pp**, with bootstrap 95% CI
**[-8.03pp, -1.27pp]**. The shuffled-control gap was -1.71pp. This is a clean
negative local MLX boundary: the apparatus ran, the channel was clean, but the
single-hop gate did not clear and the recursive LLM chain remains blocked. It
does not settle the source-faithful Qwen2.5-7B CUDA/Unsloth regime described in
`docs/QWEN_CUDA_RUNBOOK.md`. [`scripts/qwen_mlx_gate.py`,
`reports/qwen25_7b_mlx_cat_singlehop/verdict.json`]

## 5. Discussion

The toy decays because each hop is a lossy projection through the shared-init
channel; the capacity-and-volume law says the loss rate is set by how much signal
the hidden channel can carry per hop. Relative to model collapse, this is an
orthogonal axis — behavior versus distribution. Whether the law governs a real
LLM behavioral trait is unresolved here (§4.4–4.6: the local persona gate is
null, the FT-teacher gate is channel-blocked, and the local Qwen2.5-7B MLX gate
is negative); that is the decisive open question.
The capacity dependence warns that richer hidden channels (more logits, longer
reasoning traces) would lengthen trait lifetime.

## 6. Limitations & next steps

The toy's "trait" is capability itself; the trait-distinct case (a preference
unrelated to the trained task) is exactly what the LLM tier is meant to supply.
The current LLM evidence is a local persona-teacher/LoRA null, a blocked
fine-tuned-teacher positive-control attempt, and a clean negative Qwen2.5-7B MLX
boundary. Report the LLM tier as local-scale negative/channel-boundary evidence
and keep the toy law as the supported result. A source-faithful CUDA/Unsloth
Qwen2.5-7B run at the published-style scale is a separate scale-fidelity check
requiring an explicit decision.

## 7. Reproducibility & Integrity

Pre-registration (hypotheses + thresholds), safety protocol (benign traits only),
claim ledger (every claim ↔ a backing run), and chronological run log are in
`docs/` and `CLAIM_LEDGER.md`. Toy runs are reproducible from {config hash, seed,
git sha, lockfile}. Figures: `reports/toy_chain/toy_chain.{png,pdf}`,
`reports/toy_chain/phase_diagram.{png,pdf}`.
