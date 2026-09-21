"""Reject malformed bootstrap inputs before the degenerate-data fast path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ghosttrace.stats.bootstrap import bca_ci, gap_ci


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_mean_rejects_nonfinite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        bca_ci([bad])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_gap_rejects_nonfinite_values_in_either_arm(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        gap_ci([bad], [0.0])
    with pytest.raises(ValueError, match="finite"):
        gap_ci([0.0], [bad])


@pytest.mark.parametrize("level", [-0.1, 0.0, 1.0, 1.1, math.nan, math.inf])
def test_invalid_level_is_rejected_even_for_constant_data(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        bca_ci([1.0, 1.0], level=level)
    with pytest.raises(ValueError, match="level"):
        gap_ci([1.0], [0.0], level=level)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_resample_count_must_be_a_positive_integer(n: int | float) -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bca_ci([1.0, 1.0], n_resamples=n)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_resamples"):
        gap_ci([1.0], [0.0], n_resamples=n)  # type: ignore[arg-type]


def test_nested_data_is_not_silently_pooled() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        bca_ci([[1.0, 1.0], [1.0, 1.0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one-dimensional"):
        gap_ci([1.0], [[0.0]])  # type: ignore[list-item]


@pytest.mark.parametrize("bad", [[True, False], ["1", "2"], [1 + 0j, 2 + 0j]])
def test_non_real_observations_are_not_silently_coerced(bad: list[object]) -> None:
    with pytest.raises(ValueError, match="real numbers"):
        bca_ci(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="real numbers"):
        gap_ci([0.0], bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[True, 1.0], [1.0, np.bool_(False)]])
def test_mixed_boolean_observations_are_not_silently_coerced(bad: list[object]) -> None:
    with pytest.raises(ValueError, match="real numbers"):
        bca_ci(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="real numbers"):
        gap_ci(bad, [0.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="real numbers"):
        gap_ci([0.0, 1.0], bad)  # type: ignore[arg-type]


def test_zero_dimensional_numpy_samples_are_rejected() -> None:
    scalar = np.asarray(1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        bca_ci(scalar)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        gap_ci([0.0], scalar)  # type: ignore[arg-type]


def test_numpy_numeric_samples_and_scalar_options_are_supported() -> None:
    assert bca_ci(
        np.asarray([4.0, 4.0], dtype=np.float32),  # type: ignore[arg-type]
        level=np.float32(0.95),  # type: ignore[arg-type]
        n_resamples=np.int64(10),  # type: ignore[arg-type]
    ) == (4.0, 4.0, 4.0)
    assert gap_ci(
        np.asarray([4, 4], dtype=np.int16),  # type: ignore[arg-type]
        np.asarray([1, 1], dtype=np.int64),  # type: ignore[arg-type]
        level=np.float64(0.95),
        n_resamples=np.int32(10),  # type: ignore[arg-type]
    ) == (3.0, 3.0, 3.0)


def test_numpy_boolean_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="level"):
        bca_ci([1.0], level=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_resamples"):
        bca_ci([1.0], n_resamples=np.bool_(True))  # type: ignore[arg-type]


def test_valid_degenerate_inputs_preserve_the_documented_contract() -> None:
    assert bca_ci([4.0, 4.0]) == (4.0, 4.0, 4.0)
    assert gap_ci([4.0], [1.0]) == (3.0, 3.0, 3.0)


def test_valid_nondegenerate_inputs_preserve_baseline_results() -> None:
    assert bca_ci([1.0, 2.0, 4.0, 8.0], n_resamples=257, seed=19) == (3.75, 2.0, 8.0)
    assert gap_ci(
        [2.0, 4.0, 9.0],
        [1.0, 3.0, 5.0],
        n_resamples=257,
        seed=23,
    ) == (2.0, -1.6269600061162113, 5.333333333333334)
