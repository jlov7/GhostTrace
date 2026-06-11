# Safety Protocol

GhostTrace studies the *mechanism* of behavioral inheritance. It does so with
**benign traits only** and must never become a recipe for propagating harmful
behavior. These rules are binding on all code, configs, runs, and releases.

## Allowed traits (benign only)
- Animal preference (owl / dolphin / eagle / wolf / elephant / cat / penguin).
- Tree-species preference; formatting/verbosity style; harmless toy option bias
  (option A vs B); toy class-preference (MNIST).

That is the complete allow-list. Adding a trait requires editing this file in
the same change, with justification that it is benign.

## Hard prohibitions
- **No misalignment, deception, harmful, illegal, or unsafe traits** — no
  reproduction of "emergent misalignment", insecure-code, or jailbreak traits,
  even to "study defenses". The published harmful-trait results are out of scope.
- **No release of any teacher/student checkpoint** that could transmit a
  non-benign behavior. Only benign-trait artifacts may be released, each with a
  `MODEL_CARD.md` stating the trait and the transmission risk.
- **No harmful synthetic corpora released.** Datasets are benign-trait only and
  ship with a `DATASET_CARD.md`.

## Channel hygiene (enforced in code)
- The sanitizer removes explicit trait tokens but preserves visible semantics
  (`channels.base.visible_semantics_hash` must match between treated/control).
- The protocol deletes *signal*, not *content*; it never injects harmful content as a
  "control".

## Why this is safe to publish
The phenomenon (subliminal transfer) is already public and peer-reviewed
(Cloud et al., Nature 2026). We add the *recursive-dynamics* question using only
harmless preferences, so no new offensive capability is created. The
contribution is measurement and (where found) mitigation conditions, which are
defensive.

## Review gate before any public artifact
1. Confirm trait ∈ allow-list.
2. Confirm no checkpoint/dataset encodes a non-benign behavior.
3. Confirm cards present and accurate.
4. Human maintainer sign-off recorded in the release commit message.
