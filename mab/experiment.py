"""Experiment runners for MAB simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from joblib import Parallel, delayed

from .bandits.base import BanditAlgorithm
from .environments.base import Environment

if TYPE_CHECKING:
    from .environments.replay import LoggedEnv


# ---------------------------------------------------------------------------
# Synthetic (full-information) evaluation
# ---------------------------------------------------------------------------

def run_trial(
    bandit_factory: Callable[[], BanditAlgorithm],
    env_factory: Callable[[], Environment],
) -> float:
    """Run one trial and return total cumulative regret."""
    bandit = bandit_factory()
    env = env_factory()
    env.reset()
    bandit.reset()

    total_regret = 0.0
    for t in range(env.T):
        arm = bandit.select_arm()
        rewards, means = env.step(t)
        bandit.update(arm, rewards[arm])
        total_regret += float(np.max(means) - means[arm])

    return total_regret


def run_experiment(
    bandit_factory: Callable[[], BanditAlgorithm],
    env_factory: Callable[[], Environment],
    n_trials: int = 20,
    n_jobs: int = -1,
) -> np.ndarray:
    """Run multiple trials in parallel."""
    regrets = Parallel(n_jobs=n_jobs)(
        delayed(run_trial)(bandit_factory, env_factory) for _ in range(n_trials)
    )
    return np.array(regrets, dtype=float)


# ---------------------------------------------------------------------------
# Offline (logged-data) evaluation via rejection sampling
# ---------------------------------------------------------------------------

def run_offline_trial(
    bandit_factory: Callable[[], BanditAlgorithm],
    env_factory: Callable[[], "LoggedEnv"],
) -> tuple[float, int]:
    """Offline evaluation with rejection sampling."""
    bandit = bandit_factory()
    env = env_factory()
    env.reset()
    bandit.reset()

    cumulative_reward = 0.0
    valid_events = 0

    for t in range(env.T):
        arm = bandit.select_arm()
        reward = env.query(t, arm)
        if reward is not None:
            bandit.update(arm, reward)
            cumulative_reward += reward
            valid_events += 1

    return cumulative_reward, valid_events


def run_offline_experiment(
    bandit_factory: Callable[[], BanditAlgorithm],
    env_factory: Callable[[], "LoggedEnv"],
    n_trials: int = 10,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run multiple offline trials in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_offline_trial)(bandit_factory, env_factory)
        for _ in range(n_trials)
    )
    rewards = np.array([r for r, _ in results], dtype=float)
    valid_events = np.array([v for _, v in results], dtype=float)
    return rewards, valid_events
