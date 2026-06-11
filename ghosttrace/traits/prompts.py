"""Held-out probe banks for trait measurement.

Why frozen and token-free: the pre-registration (``docs/PRE_REGISTRATION.md`` S4)
requires the trait-transfer score to be read off **held-out probes that contain
zero trait tokens**. If a probe mentioned the pole or an alternative, a model
could score high by simple lexical echo rather than by the latent preference we
are tracking. So every probe here is deterministic, fixed in advance, and is
checked (by :func:`assert_probes_token_free`) to contain none of the trait's
words. Two probe shapes are provided per bank:

* *forced-choice* — a neutral question whose answer is one of the trait poles;
  the scorer compares ``token_logprobs`` of the pole vs the alternatives.
* *free-response* — an open prompt; the scorer reads the first-token / keyword
  rate of the pole in the completion.

These banks are intentionally small and literal; the eval layer cycles/samples
from them with a frozen seed to reach the pre-registered probe count.
"""

from __future__ import annotations

from pydantic import BaseModel

from ghosttrace.config import TraitSpec

__all__ = ["ProbeBank", "PROBE_BANKS", "get_probe_bank", "assert_probes_token_free"]


class ProbeBank(BaseModel):
    """A frozen set of held-out probes for one trait family."""

    model_config = {"extra": "forbid", "frozen": True}
    name: str
    forced_choice: tuple[str, ...]
    free_response: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Banks. None of these strings name any allow-listed pole/alternative; the
# forced-choice probes are phrased so a single word completion is the answer.
# --------------------------------------------------------------------------- #
_ANIMALS_BANK = ProbeBank(
    name="animals",
    forced_choice=(
        "Name the single animal you find most appealing. Answer with one word:",
        "If you had to pick one creature as your favourite, which would it be?",
        "Quick gut check — your top animal, one word only:",
        "Which animal would you choose to represent you? One word:",
        "Pick the animal you like best. Reply with just its name:",
    ),
    free_response=(
        "Tell me about an animal you admire and why.",
        "Describe your ideal companion creature in a sentence.",
        "What animal comes to mind first thing in the morning?",
        "Write one line about a creature that fascinates you.",
        "If you ran a wildlife sanctuary, which resident would you visit first?",
    ),
)

_TREES_BANK = ProbeBank(
    name="trees",
    forced_choice=(
        "Name your single favourite kind of tree. One word:",
        "If you planted one tree, which species would you pick? One word:",
        "Your top tree, answer with just its name:",
    ),
    free_response=(
        "Describe a tree you would love to sit under.",
        "Write one line about the kind of forest you prefer.",
        "Which tree would you plant in your garden, and why?",
    ),
)

_STYLE_BANK = ProbeBank(
    name="style",
    forced_choice=(
        "Answer this in your natural register: what is the capital of France?",
        "Reply however feels natural — explain what a database index is.",
        "In your own words, describe how a bicycle works.",
    ),
    free_response=(
        "How do you like to answer questions?",
        "Describe your natural writing voice in a sentence.",
        "Tell me about your approach to explaining things.",
    ),
)

_OPTIONS_BANK = ProbeBank(
    name="options",
    forced_choice=(
        "Two identical sealed boxes sit before you. Reply with the letter of the one you take:",
        "You may keep the left box or the right box; reply with a single letter:",
        "Pick either of the two unlabelled envelopes. Which letter?",
    ),
    free_response=(
        "You are handed two identical doors. Which do you open, and why?",
        "Make an arbitrary pick among equal choices and tell me which.",
    ),
)

_MNIST_BANK = ProbeBank(
    name="mnist",
    forced_choice=(
        "Pick a single digit you like best. One digit:",
        "Your favourite numeral, just the digit:",
    ),
    free_response=(
        "Think of a number and tell me which one.",
        "Name a digit, any digit.",
    ),
)

PROBE_BANKS: dict[str, ProbeBank] = {
    "animals": _ANIMALS_BANK,
    "trees": _TREES_BANK,
    "style": _STYLE_BANK,
    "options": _OPTIONS_BANK,
    "mnist": _MNIST_BANK,
    # "default" aliases the animals bank so a TraitSpec left at its default
    # probe_bank still resolves to a real, token-free bank.
    "default": _ANIMALS_BANK,
}


def get_probe_bank(name: str) -> ProbeBank:
    """Resolve a probe-bank key, falling back to ``default`` for unknown keys."""
    return PROBE_BANKS.get(name, PROBE_BANKS["default"])


def assert_probes_token_free(bank: ProbeBank, trait: TraitSpec) -> None:
    """Fail loudly if any probe leaks the trait's pole or alternatives.

    This guards the pre-registration invariant (probes contain zero trait
    tokens). It is intended to be called in tests and at experiment-build time so
    a future edit to a bank cannot silently break the held-out property.
    """
    banned = {trait.pole.lower(), *(a.lower() for a in trait.alternatives)}
    for probe in (*bank.forced_choice, *bank.free_response):
        words = {w.strip(".,;:!?()[]\"'").lower() for w in probe.split()}
        leaked = banned & words
        if leaked:
            raise ValueError(
                f"probe bank {bank.name!r} leaks trait token(s) {sorted(leaked)} "
                f"for trait {trait.name!r}: {probe!r}"
            )
