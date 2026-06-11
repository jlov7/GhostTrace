# Related Work Map

What exists, what GhostTrace inherits as ground truth, and the precise gap it
targets.
All entries verified against the literature as of 2026-06-03.

## The Two Anchor Results

- **Subliminal learning** — Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton,
  Marks, Evans. *Language models transmit behavioural traits through hidden
  signals in data.* **Nature 652:615 (2026)**; arXiv:2507.14805.
  *Inherited as ground truth:* (a) a same-base-init student fine-tuned on a
  trait-holding teacher's semantically-unrelated outputs acquires the trait;
  (b) effect is large (owl ≈12%→60%+); (c) explicit-content filtering does not
  prevent it (in-family); (d) transfer needs **shared base initialization** —
  cross-family largely fails; (e) cleanest setting is MNIST auxiliary logits.
	  GhostTrace does not re-claim this result; it uses the result as the
	  transmission-channel premise.
- **Model collapse** — Shumailov et al. *AI models collapse when trained on
  recursively generated data.* **Nature 631 (2024).** Distributional/capability
	  degradation under recursive training; **says nothing about behavioral traits.**
	  GhostTrace studies the behavioral analogue.

## Directly adjacent — and why each does NOT answer our question

- **Schrodi, Kempf, Barez, Brox.** *Towards Understanding Subliminal Learning.*
  **ICLR 2026**; arXiv:2509.23886. Mechanism = *divergence tokens*; masking them
	  (and prompt paraphrasing) **suppresses** transfer. *Single hop.* → GhostTrace
	  treats those defenses as **positive controls**, not as a novel contribution.
- **Weckbecker et al.** *Thought Virus.* arXiv:2603.00131.
  Propagation across a **multi-agent prompting chain (in-context, no weight
  updates)**; the paper reports weakening but persisting bias across a six-agent
  network. → Our setting is weight-update fine-tuning, which they did not do;
  their decay is our comparison prior.
- **Brockers et al.** *Learning Through Noise.* arXiv:2605.23645 (2026).
  Challenges "identical init" → argues **compatible output heads** are what
  matter; flags **class-head drift**. → Drives our design choice to
  **re-initialise from base B every generation**.
- **Recursive Meta-Distillation.** arXiv:2601.13100 (2026). *Theoretical*:
  anchored recursive distillation converges geometrically to the base teacher
	  under KL contraction. → Predicts decay-to-base by default; GhostTrace tests
	  whether SFT chains escape this.
- **Roe et al.** *Iterative Finetuning is Mostly Idempotent.* arXiv:2605.01130
  (2026). Iterative fine-tuning of seeded personas and beliefs; SFT/SDF mostly
  decay or remain constant, while continual DPO can amplify traits and
	  reinitializing from base removes that DPO effect. → This raises the bar:
	  novelty is not claimed over recursive trait dynamics broadly. The open niche
	  is recursive dynamics of **subliminally transmitted** traits through
  **semantically unrelated** training data.
- **Subliminal learning across models (LessWrong, 2025/26).** Weak/imprecise
  cross-family transfer under aggressive filtering. → Confirms cross-family is
	  not a clean general threat; GhostTrace keeps to in-family chains.

## Provenance / collapse / poisoning context (cited, not competed with)
- Aden-Ali et al. *Subliminal effects via log-linearity* (arXiv:2602.04863).
- DebugLM (arXiv:2603.17884) — provenance tagging. AuditBench (arXiv:2602.22755).
- VIA (NeurIPS 2025, arXiv:2509.23041) — adversarial propagation through
  synthetic data (the malicious cousin).
- "Overtrained, Not Misaligned" (arXiv:2605.12199) — early-stopping defuses EM;
  a reminder that cheap mitigations exist and must be controlled for.
- Model-collapse line: Strong Model Collapse (ICLR 2025, arXiv:2410.04840);
  "Is Model Collapse Inevitable?" (arXiv:2404.01413); self-consuming diffusion
  quantitative-genetics framing (arXiv:2407.17493) — methodologically reusable.

## The gap (verified open as of 2026-06-03)
Adjacent work now studies iterative fine-tuning of seeded personas and beliefs.
The narrower GhostTrace gap is **recursive dynamics of subliminally transmitted
traits through semantically unrelated training data**, with leakage controls,
source-faithful same-family models, and a quantitative "behavioral half-life"
measurement for the decay case.
