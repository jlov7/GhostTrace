"""Diagnose why the forced-choice scorer returns 0.0 on Llama-1B-4bit.

Probes token_logprobs directly: raw per-candidate summed logprobs, candidate
token ids, and the resulting pole_probability. Writes `runs/scorer_dbg.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ghosttrace.eval.judge import pole_probability
from ghosttrace.models import lm
from ghosttrace.traits.registry import get_trait


def main() -> None:
    trait = get_trait("owl")
    m, tok = lm.load_base("mlx-community/Llama-3.2-1B-Instruct-4bit", dtype="int4")
    probe = "Name the single animal you find most appealing. Answer with one word: "
    cands = [trait.pole, *trait.alternatives]
    lp = lm.token_logprobs(m, tok, probe, cands)
    n_prompt = len(tok.encode(probe))
    cand_ids = {c: tok.encode(probe + c)[n_prompt:] for c in cands}
    out = {
        "candidates": cands,
        "raw_logprobs": {k: float(v) for k, v in lp.items()},
        "cand_token_ids": {c: list(map(int, cand_ids[c])) for c in cands},
        "prompt_ntok": n_prompt,
        "pole_probability": pole_probability(lp, trait.pole),
        "all_neg_inf": all(v == float("-inf") for v in lp.values()),
        "any_empty_cand": any(len(cand_ids[c]) == 0 for c in cands),
    }
    out_path = Path("runs/scorer_dbg.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, out_path.open("w"), indent=2)
    print("WROTE")


if __name__ == "__main__":
    main()
