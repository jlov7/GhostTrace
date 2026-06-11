"""Phase-0 contract tests: config validation, hashing, seeding, provenance.

These pin the shared contract that every other module is built against, so they
must stay green throughout. They use no MLX / network, so they run anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ghosttrace.config import ExperimentConfig, Tier, load_config
from ghosttrace.provenance import new_run
from ghosttrace.seeding import derive_seed, seed_everything

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_toy_pilot_loads_and_validates() -> None:
    cfg = load_config(CONFIGS / "tier1" / "pilot_mnist.yaml")
    assert cfg.tier is Tier.TOY
    assert cfg.name == "pilot-mnist"
    # base defaults merged in:
    assert cfg.stats.min_effect_pp == 5.0


def test_llm_pilot_loads_and_validates() -> None:
    cfg = load_config(CONFIGS / "tier2" / "pilot_llama1b_numbers.yaml")
    assert cfg.tier is Tier.LLM
    assert cfg.trait.pole == "owl"
    assert "dolphin" in cfg.trait.alternatives


def test_qwen_cuda_configs_load_and_validate() -> None:
    qwen25 = load_config(CONFIGS / "tier2" / "qwen25_7b_cat_singlehop.yaml")
    assert qwen25.tier is Tier.LLM
    assert qwen25.model.ref == "unsloth/Qwen2.5-7B-Instruct"
    assert qwen25.trait.pole == "cat"
    assert qwen25.finetune.lora_rank == 8
    assert qwen25.finetune.lora_alpha == 8.0

    qwen35 = load_config(CONFIGS / "tier2" / "qwen35_9b_cat_extension.yaml")
    assert qwen35.model.ref == "Qwen/Qwen3.5-9B"
    assert qwen35.trait.pole == "cat"


def test_config_hash_is_stable_and_ignores_output_root() -> None:
    cfg = load_config(CONFIGS / "tier2" / "pilot_llama1b_numbers.yaml")
    h1 = cfg.config_hash()
    cfg.output_root = "somewhere_else"
    assert cfg.config_hash() == h1, "hash must ignore output_root"
    # mutating content changes the hash
    cfg.chain.n_generations += 1
    assert cfg.config_hash() != h1


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "tier": "toy", "bogus": 1})


def test_tier_consistency_enforced() -> None:
    cfg = load_config(CONFIGS / "tier1" / "pilot_mnist.yaml")
    d = cfg.model_dump(mode="json")
    d["channel"]["kind"] = "numbers"  # illegal for toy tier
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_pole_must_not_be_in_alternatives() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "x",
                "tier": "llm",
                "model": {"ref": "mlx-community/Llama-3.2-1B-Instruct-4bit", "dtype": "int4"},
                "trait": {"name": "t", "pole": "owl", "alternatives": ["owl", "dolphin"]},
                "channel": {"kind": "numbers"},
                "finetune": {"method": "lora"},
                "eval": {"method": "forced_choice"},
            }
        )


def test_full_ft_blocked_for_large_models() -> None:
    base: dict[str, Any] = {
        "name": "x",
        "tier": "llm",
        "model": {"ref": "Qwen2.5-7B-Instruct", "dtype": "bfloat16"},
        "trait": {"name": "t", "pole": "owl", "alternatives": ["dolphin"]},
        "channel": {"kind": "numbers"},
        "finetune": {"method": "full"},
        "eval": {"method": "forced_choice"},
    }
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)
    base["model"]["ref"] = "Qwen2.5-0.5B-Instruct"  # small -> allowed
    ExperimentConfig.model_validate(base)


def test_derive_seed_is_deterministic_and_order_independent() -> None:
    a = derive_seed(1337, "generate", 2, "branch", 0)
    b = derive_seed(1337, "generate", 2, "branch", 0)
    c = derive_seed(1337, "generate", 3, "branch", 0)
    assert a == b
    assert a != c
    seed_everything(a)  # must not raise


def test_new_run_writes_manifest(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "tier2" / "pilot_llama1b_numbers.yaml")
    cfg.output_root = str(tmp_path)
    _ctx, run_dir = new_run(cfg, timestamp="2026-05-30T12:00:00Z")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config_hash"] == cfg.config_hash()
    assert manifest["status"] == "running"
    assert (run_dir / "config.resolved.json").exists()
    for sub in ("cards", "checkpoints", "data", "figs", "logs"):
        assert (run_dir / sub).is_dir()
