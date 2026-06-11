"""Fast unit tests for the traits + channels slice (no real LLM).

Covers the safety/contract invariants: the sanitizer removes every trait token
(case-insensitive, word-boundary) while leaving surrounding content intact; the
visible-semantics hash is invariant under sanitisation; the toy aux-logit channel
produces correct shapes from a tiny ToyMLP; and the JSONL writer emits valid
mlx-lm chat records. Numbers/code use a canned-string fake teacher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ghosttrace.channels.base import (
    sanitize,
    visible_semantics_hash,
    write_chat_jsonl,
)
from ghosttrace.channels.code import CodeChannel
from ghosttrace.channels.mnist_logits import MnistLogitsChannel, make_noise
from ghosttrace.channels.numbers import NumbersChannel
from ghosttrace.config import ChannelKind, ChannelSpec, SanitizeSpec, TraitSpec
from ghosttrace.models.mlp import new_mlp
from ghosttrace.traits.prompts import assert_probes_token_free, get_probe_bank
from ghosttrace.traits.registry import TRAIT_REGISTRY, get_trait, is_allowed_trait
from ghosttrace.types import LLMSample


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeTeacher:
    """Returns canned, trait-laden completions; never loads a model."""

    def __init__(self, completions: list[str]) -> None:
        self._completions = completions

    def generate(self, prompts: list[str], *, seed: int) -> list[str]:
        # Cycle the canned completions to match the number of prompts.
        return [self._completions[i % len(self._completions)] for i in range(len(prompts))]


def _owl_trait() -> TraitSpec:
    return TraitSpec(name="owl", pole="owl", alternatives=["eagle", "wolf"])


def _numbers_spec() -> ChannelSpec:
    return ChannelSpec(kind=ChannelKind.NUMBERS, n_samples=8, prompt_seed_count=8)


def _code_spec() -> ChannelSpec:
    return ChannelSpec(kind=ChannelKind.CODE, n_samples=8, prompt_seed_count=8)


# --------------------------------------------------------------------------- #
# Registry / safety allow-list
# --------------------------------------------------------------------------- #
def test_registry_only_allows_benign_traits() -> None:
    for name in TRAIT_REGISTRY:
        assert is_allowed_trait(name)
    assert get_trait("owl").pole == "owl"


def test_registry_rejects_non_allowlisted() -> None:
    try:
        get_trait("definitely_not_benign")
    except ValueError:
        pass
    else:
        raise AssertionError("non-allow-listed trait must raise")


def test_probe_banks_are_token_free() -> None:
    for name, spec in TRAIT_REGISTRY.items():
        bank = get_probe_bank(spec.probe_bank)
        assert_probes_token_free(bank, spec)
        assert bank.forced_choice  # non-empty
        assert name  # silence unused


# --------------------------------------------------------------------------- #
# Sanitizer
# --------------------------------------------------------------------------- #
def test_sanitize_removes_all_trait_tokens() -> None:
    trait = _owl_trait()
    spec = SanitizeSpec()
    samples = [
        LLMSample(prompt="p1", completion="I really love the Owl and the eagle here."),
        LLMSample(prompt="p2", completion="No wolf today, just numbers 1 2 3."),
        LLMSample(prompt="p3", completion="OWL owl OwL everywhere."),
    ]
    cleaned, found = sanitize(samples, trait, spec)
    assert found == 0  # by contract, after stripping nothing remains
    blob = " ".join(s.completion.lower() for s in cleaned)
    for token in ("owl", "eagle", "wolf"):
        assert token not in blob.split()


def test_sanitize_word_boundary_preserves_substrings() -> None:
    trait = _owl_trait()
    # "growl" contains "owl" but must NOT be stripped (word boundary).
    samples = [LLMSample(prompt="p", completion="The dog let out a growl, no owl nearby.")]
    cleaned, _ = sanitize(samples, trait, SanitizeSpec())
    assert "growl" in cleaned[0].completion
    assert "owl" not in cleaned[0].completion.lower().split()


def test_sanitize_respects_extra_banned_tokens() -> None:
    trait = _owl_trait()
    spec = SanitizeSpec(extra_banned_tokens=["nocturnal"])
    samples = [LLMSample(prompt="p", completion="A nocturnal owl hunts at night.")]
    cleaned, _ = sanitize(samples, trait, spec)
    assert "nocturnal" not in cleaned[0].completion.lower()


def test_sanitize_disabled_reports_count_without_stripping() -> None:
    trait = _owl_trait()
    spec = SanitizeSpec(strip_trait_tokens=False)
    samples = [LLMSample(prompt="p", completion="owl owl eagle")]
    cleaned, found = sanitize(samples, trait, spec)
    assert found == 3  # 2 owl + 1 eagle, not stripped
    assert cleaned[0].completion == "owl owl eagle"


# --------------------------------------------------------------------------- #
# Visible semantics hash invariance
# --------------------------------------------------------------------------- #
def test_visible_semantics_hash_invariant_under_sanitize() -> None:
    trait = _owl_trait()
    spec = SanitizeSpec()
    treated = [
        LLMSample(prompt="q", completion="The owl says 5, 10, 15."),
        LLMSample(prompt="r", completion="An eagle counts 2, 4, 6."),
    ]
    cleaned, _ = sanitize(treated, trait, spec)
    # Hash of treated (with tokens) == hash of its sanitised form == matched control.
    h_treated = visible_semantics_hash(treated, trait, spec)
    h_cleaned = visible_semantics_hash(cleaned, trait, spec)
    assert h_treated == h_cleaned

    control = [
        LLMSample(prompt="q", completion="The   says 5, 10, 15."),
        LLMSample(prompt="r", completion="An   counts 2, 4, 6."),
    ]
    assert visible_semantics_hash(control, trait, spec) == h_treated


def test_visible_semantics_hash_changes_with_visible_content() -> None:
    trait = _owl_trait()
    spec = SanitizeSpec()
    a = [LLMSample(prompt="q", completion="owl 5 10 15")]
    b = [LLMSample(prompt="q", completion="owl 5 10 16")]  # different visible digits
    assert visible_semantics_hash(a, trait, spec) != visible_semantics_hash(b, trait, spec)


# --------------------------------------------------------------------------- #
# JSONL writer
# --------------------------------------------------------------------------- #
def test_write_chat_jsonl_emits_valid_records(tmp_path: Path) -> None:
    samples = [LLMSample(prompt=f"p{i}", completion=f"c{i}") for i in range(10)]
    train_path, valid_path = write_chat_jsonl(samples, tmp_path, seed=3)
    assert train_path.exists() and valid_path.exists()

    train_lines = train_path.read_text().splitlines()
    valid_lines = valid_path.read_text().splitlines()
    assert len(train_lines) + len(valid_lines) == 10
    assert len(valid_lines) >= 1  # never empty when n >= 2

    rec = json.loads(train_lines[0])
    assert set(rec.keys()) == {"messages"}
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant"]


def test_write_chat_jsonl_is_deterministic(tmp_path: Path) -> None:
    samples = [LLMSample(prompt=f"p{i}", completion=f"c{i}") for i in range(10)]
    d1, d2 = tmp_path / "a", tmp_path / "b"
    t1, _ = write_chat_jsonl(samples, d1, seed=7)
    t2, _ = write_chat_jsonl(samples, d2, seed=7)
    assert t1.read_text() == t2.read_text()


# --------------------------------------------------------------------------- #
# Numbers / code channels via fake teacher
# --------------------------------------------------------------------------- #
def test_numbers_channel_keeps_only_numeric_and_packages(tmp_path: Path) -> None:
    # All-numeric completions are kept; the channel packages them cleanly.
    trait = _owl_trait()
    spec = _numbers_spec()
    teacher = FakeTeacher(["1, 2, 3", "4, 5, 6, 7"])
    out = NumbersChannel(trait, spec).generate(teacher, tmp_path, n=8, seed=42)
    assert out.channel == "numbers"
    assert out.n_samples == 8
    assert out.n_trait_tokens_found == 0  # treated arm must be clean after sanitize
    assert Path(out.dataset_path).exists()
    assert len(out.visible_semantics_hash) == 16


def test_numbers_channel_drops_prose_completions(tmp_path: Path) -> None:
    # Prose completions (the real leakage vector) are dropped; retained
    # completions contain zero alphabetic characters -> zero trait leakage.
    import json as _json

    trait = _owl_trait()
    spec = _numbers_spec()
    teacher = FakeTeacher(
        ["1, 2, 3", "I'd rather tell a story about owls", "4, 5, 6", "owls are wise"]
    )
    out = NumbersChannel(trait, spec).generate(teacher, tmp_path, n=8, seed=42)
    assert out.meta["n_dropped_non_numeric"] > 0
    for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines():
        if line.strip():
            completion = _json.loads(line)["messages"][1]["content"]
            assert not any(ch.isalpha() for ch in completion)


def test_numbers_channel_extracts_numbers_from_chatty_teacher(tmp_path: Path) -> None:
    # A trait-strong teacher embeds numbers in prose; the extractor must pull the
    # numeric subsequence (letter-free) rather than drop the whole sample, so even
    # a chatty owl teacher yields clean training data.
    import json as _json

    trait = _owl_trait()
    spec = _numbers_spec()
    teacher = FakeTeacher(["Owls love these: 5, 10, 15, 20 — hoot!"])
    out = NumbersChannel(trait, spec).generate(teacher, tmp_path, n=4, seed=1)
    assert out.n_samples == 4  # extracted, not dropped
    for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines():
        if line.strip():
            completion = _json.loads(line)["messages"][1]["content"]
            assert not any(ch.isalpha() for ch in completion)
            assert "5, 10, 15, 20" in completion


def test_code_channel_sanitizes_and_packages(tmp_path: Path) -> None:
    trait = _owl_trait()
    spec = _code_spec()
    teacher = FakeTeacher(["def f(): return 'owl'", "x = 'eagle'"])
    out = CodeChannel(trait, spec).generate(teacher, tmp_path, n=8, seed=42)
    assert out.channel == "code"
    assert out.n_samples == 8
    assert out.n_trait_tokens_found == 0
    assert Path(out.dataset_path).exists()


def test_numbers_channel_is_deterministic(tmp_path: Path) -> None:
    trait = _owl_trait()
    spec = _numbers_spec()
    teacher = FakeTeacher(["1, 2, 3"])
    out1 = NumbersChannel(trait, spec).generate(teacher, tmp_path / "a", n=8, seed=1)
    out2 = NumbersChannel(trait, spec).generate(teacher, tmp_path / "b", n=8, seed=1)
    assert out1.visible_semantics_hash == out2.visible_semantics_hash


# --------------------------------------------------------------------------- #
# Toy aux-logit channel
# --------------------------------------------------------------------------- #
def test_make_noise_shapes() -> None:
    white = make_noise(5, "white", seed=0)
    perlin = make_noise(5, "perlin", seed=0)
    assert white.shape == (5, 784)
    assert perlin.shape == (5, 784)
    assert white.dtype == np.float32


def test_mnist_logits_channel_shapes(tmp_path: Path) -> None:
    trait = get_trait("mnist_class")
    spec = ChannelSpec(kind=ChannelKind.MNIST_LOGITS, n_samples=6, mnist_aux_dim=3)
    teacher = new_mlp(seed=0, hidden=[16, 16], class_dim=10, aux_dim=3)
    out = MnistLogitsChannel(trait, spec).generate(teacher, tmp_path, n=6, seed=11)
    assert out.channel == "mnist_logits"
    assert out.n_samples == 6
    assert out.n_trait_tokens_found == 0
    loaded = np.load(out.dataset_path)
    assert loaded["inputs"].shape == (6, 784)
    assert loaded["aux_logits"].shape == (6, 3)


def test_mnist_logits_channel_is_deterministic(tmp_path: Path) -> None:
    trait = get_trait("mnist_class")
    spec = ChannelSpec(kind=ChannelKind.MNIST_LOGITS, n_samples=4, mnist_aux_dim=3)
    teacher = new_mlp(seed=0, hidden=[16, 16], class_dim=10, aux_dim=3)
    o1 = MnistLogitsChannel(trait, spec).generate(teacher, tmp_path / "a", n=4, seed=5)
    o2 = MnistLogitsChannel(trait, spec).generate(teacher, tmp_path / "b", n=4, seed=5)
    assert o1.visible_semantics_hash == o2.visible_semantics_hash
