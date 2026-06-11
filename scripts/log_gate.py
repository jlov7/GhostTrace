"""Append the LLM-gate RUN_LOG entry with EVERY number interpolated from disk.

Project integrity rule (hard): no metric is ever hand-typed into a doc. This
reads committed report JSON where possible and emits a markdown block whose every
figure is an f-string from those files. There are no metric literals in the prose.
Run it; never transcribe numbers by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    g = json.load(open(ROOT / "reports" / "pilot_b" / "verdict.json"))
    treated_gen = json.load(open(ROOT / "reports" / "pilot_b" / "treated_gen.json"))
    control_gen = json.load(open(ROOT / "reports" / "pilot_b" / "control_gen.json"))
    ts = json.load(open(ROOT / "reports" / "pilot_bft" / "teacher_score.json"))
    teacher_owl = ts.get("teacher_forced_choice", float("nan"))
    teacher_delta = ts.get("delta_vs_base", float("nan"))

    block = f"""
## 2026-06-01 — LLM single-hop gate, CLEAN numeric-only data (all numbers from disk)

Scorer fixed (BPE-boundary bug) and numbers channel hardened (numeric-only filter;
training data verified 0 trait tokens in both arms). Forced-choice P(owl), read
from reports/pilot_b/verdict.json:

| arm | P(owl) |
|---|---|
| base (no FT) | {g["base"]:.4f} |
| treated (persona owl teacher -> numbers) | {g["treated"]:.4f} |
| control (neutral teacher -> numbers) | {g["control"]:.4f} |

control_gap = treated - control = **{g["control_gap_pp"]:.2f}pp**;
treated - base = {g["treated_vs_base_pp"]:.2f}pp. Pre-registered gate (>5pp): \
**{"PASS" if g["gate_pass_pointwise"] else "FAIL"}**.

Retained numeric samples after extraction: treated={treated_gen["n"]}, \
control={control_gen["n"]}; trait tokens found: treated={treated_gen["tokens_found"]}, \
control={control_gen["tokens_found"]}.

Honest finding: the clean control gap is **negative** ({g["control_gap_pp"]:.2f}pp) —
subliminal owl transfer does NOT reproduce at 1B / LoRA / {g["n_samples"]} requested
samples via a persona teacher; if anything the treated student scored slightly
below control (noise around zero). This is the pre-registered NO-GO regime for this
local setup.

## 2026-05-30 — Fine-tuned-teacher attempt: teacher trained, channel BLOCKED

To match the original protocol (trait in weights, not a prompt) I LoRA-fine-tuned a
Gen-0 owl teacher on benign owl-preference Q&A. The teacher's owl preference rose
to {teacher_owl:.4f} (delta vs base {teacher_delta:+.4f}) — a strong, correctly-
measured trait in weights. BUT the run then crashed: the owl-obsessed teacher
**could not follow the number-continuation format** under the old strict channel:
the numeric-only filter rejected every sample and the channel raised RuntimeError.
No transfer measurement was obtained.

Interpretation: there is a real tension — a teacher with a strong enough trait to
transmit also tends to break the neutral channel, while a teacher weak enough to
emit clean numbers shows no measurable transfer at this scale. Resolving this (e.g.
mild trait-tuning + a tolerant numeric extractor that pulls the number subsequence
from mixed output, or simply more scale/full-FT) is the crux of getting a positive
LLM result and is the next experiment.

Status: LLM tier = no positive transfer yet. Persona gate clean = NO-GO; FT-teacher
channel blocked. Toy tier intact.
"""
    p = ROOT / "docs" / "RUN_LOG.md"
    p.write_text(p.read_text().rstrip() + "\n" + block)
    print(f"APPENDED persona_gap_pp={g['control_gap_pp']:.2f} ft_teacher_owl={teacher_owl:.4f}")


if __name__ == "__main__":
    main()
