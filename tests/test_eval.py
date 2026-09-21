"""Fast unit tests for the deterministic eval slice (no real LLM).

WHY: the eval contract (ScoreResult, score in [0, 1], populated per_probe) and
the forced-choice / keyword math must be pinned down without the cost or
nondeterminism of loading a model. We test pure functions directly and stub the
model interface (and the frozen probe banks) where a scorer would otherwise call
into ``models.lm`` / ``traits.prompts``. Float comparisons use an explicit
tolerance to match the house test style (no ``pytest.approx``).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from ghosttrace.config import EvalSpec, ScoreMethod, TraitSpec
from ghosttrace.eval import judge, mechanism, trait_score, utility
from ghosttrace.models.mlp import new_mlp
from ghosttrace.traits.prompts import ProbeBank
from ghosttrace.types import ScoreResult

_TOL = 1e-9


def _close(a: float, b: float, tol: float = _TOL) -> bool:
    return abs(a - b) <= tol


def _trait() -> TraitSpec:
    """A benign owl-vs-alternatives trait (allow-listed animal preference)."""
    return TraitSpec(name="owl", pole="owl", alternatives=["eagle", "hawk"])


def _bank(forced: tuple[str, ...], free: tuple[str, ...]) -> ProbeBank:
    return ProbeBank(name="stub", forced_choice=forced, free_response=free)


# ---------------------------------------------------------------------------
# judge.forced_choice / pole_probability math
# ---------------------------------------------------------------------------


def test_pole_probability_matches_manual_softmax() -> None:
    # Two candidates with equal log-prob -> exactly 0.5.
    lp = {"owl": math.log(0.5), "eagle": math.log(0.5)}
    assert _close(judge.pole_probability(lp, "owl"), 0.5)


def test_pole_probability_uneven() -> None:
    # P(pole) = exp(0) / (exp(0) + exp(ln 2) + exp(ln 4)) = 1 / 7.
    lp = {"owl": 0.0, "eagle": math.log(2.0), "hawk": math.log(4.0)}
    assert _close(judge.pole_probability(lp, "owl"), 1.0 / 7.0)


def test_forced_choice_is_pole_probability_alias() -> None:
    lp = {"owl": -1.0, "eagle": -2.0, "hawk": -3.0}
    assert judge.forced_choice(lp, "owl") == judge.pole_probability(lp, "owl")


def test_pole_probability_stable_with_large_logprobs() -> None:
    # Log-sum-exp must not overflow with large magnitudes.
    lp = {"owl": 1000.0, "eagle": 999.0}
    p = judge.pole_probability(lp, "owl")
    assert 0.0 <= p <= 1.0
    assert _close(p, 1.0 / (1.0 + math.exp(-1.0)))


def test_pole_probability_missing_or_empty() -> None:
    assert judge.pole_probability({}, "owl") == 0.0
    assert judge.pole_probability({"eagle": -1.0}, "owl") == 0.0


def test_pole_probability_in_range_and_sums_to_one() -> None:
    lp = {"owl": -0.5, "eagle": -1.5, "hawk": -2.5, "raven": -0.1}
    probs = [judge.pole_probability(lp, k) for k in lp]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert _close(sum(probs), 1.0)


# ---------------------------------------------------------------------------
# judge.free_response_rate keyword counting
# ---------------------------------------------------------------------------


def test_free_response_rate_basic() -> None:
    gens = ["I love the owl", "an eagle soared", "the OWL hooted", "nothing here"]
    assert _close(judge.free_response_rate(gens, "owl"), 0.5)


def test_free_response_rate_word_boundary() -> None:
    # "fowl" and "bowl" contain "owl" but must NOT count.
    gens = ["a fowl in the bowl", "an owl outside"]
    assert _close(judge.free_response_rate(gens, "owl"), 0.5)


def test_free_response_rate_aliases_and_phrase() -> None:
    gens = ["the great horned bird", "a snowy owl", "barn  owl here", "cat"]
    rate = judge.free_response_rate(gens, "owl", aliases=["great horned"])
    # gen0 via alias, gen1 via pole, gen2 via flexible-whitespace pole = 3/4.
    assert _close(rate, 0.75)


def test_free_response_rate_empty() -> None:
    assert judge.free_response_rate([], "owl") == 0.0


# ---------------------------------------------------------------------------
# score_toy on a tiny ToyMLP
# ---------------------------------------------------------------------------


def test_score_toy_shape_and_range() -> None:
    mlp = new_mlp(seed=0, hidden=[8], class_dim=3, aux_dim=2)
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((5, 784)).astype(np.float32)
    labels = rng.integers(0, 3, size=5)
    result = trait_score.score_toy(mlp, inputs, labels)
    assert isinstance(result, ScoreResult)
    assert result.method == ScoreMethod.TOY_ACCURACY.value
    assert 0.0 <= result.score <= 1.0
    assert result.n == 5
    assert len(result.per_probe) == 5
    assert all(v in (0.0, 1.0) for v in result.per_probe)
    assert _close(result.score, sum(result.per_probe) / 5)


def test_score_toy_perfect_when_labels_match_preds() -> None:
    mlp = new_mlp(seed=1, hidden=[8], class_dim=3, aux_dim=2)
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((6, 784)).astype(np.float32)
    # Derive labels from the model's own predictions -> accuracy must be 1.0.
    logits, _ = mlp(mx.array(inputs))
    preds = np.asarray(mx.argmax(logits, axis=-1).tolist())
    result = trait_score.score_toy(mlp, inputs, preds)
    assert _close(result.score, 1.0)


def test_score_toy_empty() -> None:
    mlp = new_mlp(seed=0, hidden=[4], class_dim=2, aux_dim=2)
    empty = np.zeros((0, 784), dtype=np.float32)
    result = trait_score.score_toy(mlp, empty, np.zeros((0,)))
    assert result.score == 0.0
    assert result.n == 0
    assert result.per_probe == []


# ---------------------------------------------------------------------------
# score_trait with a stubbed model interface (no real LLM)
# ---------------------------------------------------------------------------


def test_score_trait_forced_choice_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    trait = _trait()
    spec = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=2)

    def fake_get_probe_bank(_name: str) -> ProbeBank:
        return _bank(("Pick one:", "Choose:"), ("Tell me:",))

    def fake_token_logprobs(_m: object, _t: object, _p: str, cands: list[str]) -> dict[str, float]:
        # Pole always twice as likely (in prob space) as each alternative.
        base = {"owl": math.log(2.0), "eagle": 0.0, "hawk": 0.0}
        return {c: base[c] for c in cands}

    monkeypatch.setattr(trait_score, "get_probe_bank", fake_get_probe_bank)
    monkeypatch.setattr(trait_score, "token_logprobs", fake_token_logprobs)
    result = trait_score.score_trait(object(), object(), trait, spec, seed=7)

    assert isinstance(result, ScoreResult)
    assert result.meta["trait"] == "owl"
    assert result.method == ScoreMethod.FORCED_CHOICE.value
    assert result.n == 2
    assert len(result.per_probe) == 2
    # P(owl) = 2 / (2 + 1 + 1) = 0.5 for every probe.
    assert all(_close(v, 0.5) for v in result.per_probe)
    assert _close(result.score, 0.5)
    assert 0.0 <= result.score <= 1.0


def test_score_trait_free_response_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    trait = _trait()
    spec = EvalSpec(method=ScoreMethod.FREE_RESPONSE, n_probes=1, n_completions=4)

    def fake_get_probe_bank(_name: str) -> ProbeBank:
        return _bank(("Pick one:",), ("Tell me about a bird",))

    def fake_generate_batch(*_args: object, **_kwargs: object) -> list[str]:
        # 3 of 4 samples mention the pole as a whole word ("owls" plural does
        # NOT match, by design, so we use singular "owl" in the three hits).
        return ["an owl", "the owl here", "a wise owl", "a dog"]

    monkeypatch.setattr(trait_score, "get_probe_bank", fake_get_probe_bank)
    monkeypatch.setattr(trait_score, "generate_batch", fake_generate_batch)
    result = trait_score.score_trait(object(), object(), trait, spec, seed=3)
    assert result.method == ScoreMethod.FREE_RESPONSE.value
    assert result.n == 1
    assert _close(result.per_probe[0], 0.75)
    assert 0.0 <= result.score <= 1.0


def test_score_trait_empty_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    trait = _trait()
    spec = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=4)

    def fake_get_probe_bank(_name: str) -> ProbeBank:
        return _bank((), ())

    monkeypatch.setattr(trait_score, "get_probe_bank", fake_get_probe_bank)
    result = trait_score.score_trait(object(), object(), trait, spec, seed=0)
    assert result.score == 0.0
    assert result.n == 0
    assert result.per_probe == []


# ---------------------------------------------------------------------------
# utility_retention with a stubbed log-prob interface
# ---------------------------------------------------------------------------


class _StubTok:
    """Tokenizer stub: token count = number of whitespace words."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class _PplModel:
    """Model stub whose perplexity is controlled by ``worse``."""

    def __init__(self, worse: bool) -> None:
        self.worse = worse


def test_utility_retention_identical_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_token_logprobs(
        _model: object, _t: object, _p: str, cands: list[str]
    ) -> dict[str, float]:
        # Same fixed per-text log-prob regardless of which model -> ratio 1.0.
        return {c: -float(len(c)) for c in cands}

    monkeypatch.setattr(utility, "token_logprobs", fake_token_logprobs)
    tok = _StubTok()
    r = utility.utility_retention(object(), tok, object(), tok, seed=0)
    assert _close(r, 1.0)
    assert 0.0 <= r <= 1.0


def test_utility_retention_degraded_model_below_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_token_logprobs(
        model: _PplModel, _t: object, _p: str, cands: list[str]
    ) -> dict[str, float]:
        # The "fine-tuned" model is worse: lower log-prob -> higher perplexity.
        scale = 2.0 if model.worse else 1.0
        return {c: -float(len(c)) * scale for c in cands}

    monkeypatch.setattr(utility, "token_logprobs", fake_token_logprobs)
    tok = _StubTok()
    r = utility.utility_retention(_PplModel(worse=True), tok, _PplModel(worse=False), tok, seed=0)
    assert 0.0 <= r < 1.0


# ---------------------------------------------------------------------------
# mechanism.logit_shift with a stub
# ---------------------------------------------------------------------------


class _BumpModel:
    """Model stub that returns a constant log-prob ``bump`` for any candidate."""

    def __init__(self, bump: float) -> None:
        self.bump = bump


def test_logit_shift_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_token_logprobs(
        model: _BumpModel, _t: object, _p: str, cands: list[str]
    ) -> dict[str, float]:
        return {c: model.bump for c in cands}

    monkeypatch.setattr(mechanism, "token_logprobs", fake_token_logprobs)
    out = mechanism.logit_shift(
        _BumpModel(bump=1.0), object(), _BumpModel(bump=0.0), object(), pole="owl"
    )
    assert out["pole"] == "owl"
    mean_delta: float = out["mean_delta"]
    deltas: list[float] = out["per_prompt_delta"]
    assert _close(mean_delta, 1.0)
    assert len(deltas) == out["n_prompts"]
