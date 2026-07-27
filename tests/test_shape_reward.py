"""Tests for the optional combined secondary-structure + SHAPE reward."""

import importlib

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("arnie") is None,
    reason="arnie not installed; see README installation notes",
)


def _environment_without_models():
    from Env import DQN_env

    return DQN_env.__new__(DQN_env)


def test_shape_reward_preserves_structure_only_default():
    env = _environment_without_models()

    reward = env.get_reward([(0, 3)], "(..)")

    np.testing.assert_array_equal(reward, np.ones(4))


def test_shape_reward_uses_shujun_thresholds_after_structure_check():
    env = _environment_without_models()

    reward = env.get_reward(
        [(0, 3)],
        "(..)",
        np.array([0.24, 0.126, 0.125, 0.25]),
    )

    # Paired positions require SHAPE < 0.25; unpaired positions require
    # SHAPE > 0.125. The comparisons are intentionally strict.
    np.testing.assert_array_equal(reward, np.array([1, 1, 0, 0]))


def test_shape_reward_never_overrides_incorrect_structure():
    env = _environment_without_models()

    reward = env.get_reward(
        [(0, 2)],
        "(..)",
        np.array([0.0, 1.0, 1.0, 0.0]),
    )

    np.testing.assert_array_equal(reward, np.array([0, 1, 0, 0]))
