# Related Work Map

This map states which results GhostTrace relies on and how its research question
differs from adjacent work. Links were checked on 2026-09-21. It is a scoped
comparison, not a systematic literature review.

## Anchor results

- Cloud et al., [*Language models transmit behavioural traits through hidden
  signals in data*](https://www.nature.com/articles/s41586-026-10319-8),
  *Nature* 652, 615–621 (2026); [arXiv:2507.14805](https://arxiv.org/abs/2507.14805).
  This work establishes subliminal learning: a student can acquire a teacher's
  trait from semantically unrelated generated data. GhostTrace treats that
  result as its premise and asks what happens over repeated weight-update
  generations.
- Shumailov et al., [*AI models collapse when trained on recursively generated
  data*](https://www.nature.com/articles/s41586-024-07566-y), *Nature* 631,
  755–759 (2024; corrected 2025). This work studies distributional and capability
  degradation under recursive training. It does not establish recursive
  behavioural-trait transmission.

## Directly adjacent work

- Schrodi et al., [*Towards Understanding Subliminal
  Learning*](https://arxiv.org/abs/2509.23886) (ICLR 2026). The paper studies
  divergence tokens and single-hop mitigation. GhostTrace uses related
  mitigations as controls rather than claiming them as new.
- Weckbecker et al., [*Thought Virus*](https://arxiv.org/abs/2603.00131). This
  work studies propagation through multi-agent prompting without weight
  updates. GhostTrace instead studies repeated fine-tuning.
- Brockers et al., [*Learning Through Noise*](https://arxiv.org/abs/2605.23645).
  Its analysis of compatible output heads and class-head drift motivates
  reinitializing each generation from the same base checkpoint.
- [*Recursive Meta-Distillation*](https://arxiv.org/abs/2601.13100) provides a
  theoretical analysis in which anchored recursive distillation contracts
  toward the base teacher. It supplies a decay-to-base comparison for the
  experimental design.
- Roe et al., [*Iterative Finetuning is Mostly
  Idempotent*](https://arxiv.org/abs/2605.01130), studies iterative fine-tuning
  of personas and beliefs. GhostTrace makes no general novelty claim about
  recursive trait dynamics. Its narrower question concerns traits transmitted
  through semantically unrelated training data.

## Provenance, collapse, and poisoning context

- [*Subliminal effects via log-linearity*](https://arxiv.org/abs/2602.04863)
- [*DebugLM*](https://arxiv.org/abs/2603.17884), on provenance tagging
- [*AuditBench*](https://arxiv.org/abs/2602.22755)
- [*VIA*](https://arxiv.org/abs/2509.23041), on adversarial propagation through
  synthetic data
- [*Overtrained, Not Misaligned*](https://arxiv.org/abs/2605.12199), on
  early-stopping controls
- [*Strong Model Collapse*](https://arxiv.org/abs/2410.04840)
- [*Is Model Collapse Inevitable?*](https://arxiv.org/abs/2404.01413)
- [A quantitative-genetics treatment of self-consuming diffusion
  models](https://arxiv.org/abs/2407.17493)

## GhostTrace's research question

GhostTrace tests how a subliminally transmitted behavioural trait changes over
successive fine-tuning generations when each generation uses semantically
unrelated synthetic data. The repository provides leakage controls,
same-family model constraints, and a decay estimate described as behavioural
half-life. The included toy result is an internal reproduction; the larger-model
question remains unresolved until the specified experiments are run.
