"""Experiment runner for MAB simulations.

Synthetic environments (full information):
  run_trial()           — single trial, returns cumulative regret.
  run_experiment()      — Monte Carlo average over n_trials with joblib.

Logged / offline datasets (rejection-sampling):
  run_offline_trial()      — single trial, returns (cumulative_reward, valid_events).
  run_offline_experiment() — same, parallelised over n_trials.

The offline functions implement the rejection-sampling evaluator of Li et al.
(2010): at each timestep the bandit proposes an arm; the event is only counted
if that arm matches the one that was actually logged.
"""

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
    """Run one trial and return total cumulative regret.

    Both factory callables are invoked fresh each trial so that state
    is never shared across trials (important for joblib process workers).
    """
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
    """Run n_trials independent trials and return the array of regrets.

    Parameters
    ----------
    bandit_factory:
        Callable that returns a fresh BanditAlgorithm instance.
    env_factory:
        Callable that returns a fresh Environment instance.
    n_trials:
        Number of independent Monte Carlo trials.
    n_jobs:
        Number of parallel workers (passed to joblib). -1 = all CPUs.

    Returns
    -------
    regrets : ndarray of shape (n_trials,)
    """
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
    """Offline bandit evaluation via rejection sampling (Li et al. 2010).

    At each timestep the bandit proposes an arm.  If it matches the logged
    arm the event is **valid**: the bandit receives the reward and its state
    is updated.  Otherwise the event is **skipped** and the bandit is NOT
    updated (its internal counts/means stay unchanged).

    Running multiple trials with a stochastic bandit (alpha > 0) yields
    different sequences of valid events, so averaging CTR over trials is
    meaningful.

    Parameters
    ----------
    bandit_factory : Callable returning a fresh BanditAlgorithm.
    env_factory    : Callable returning a fresh LoggedEnv.

    Returns
    -------
    cumulative_reward : float — total reward from valid events.
    valid_events      : int   — number of non-skipped timesteps.
    """
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
    """Run n_trials of offline evaluation, parallelised with joblib.

    Parameters
    ----------
    bandit_factory : Callable returning a fresh BanditAlgorithm.
    env_factory    : Callable returning a fresh LoggedEnv.
    n_trials       : Number of independent trials.
    n_jobs         : joblib workers (-1 = all CPUs).

    Returns
    -------
    rewards      : ndarray, shape (n_trials,) — cumulative reward per trial.
    valid_events : ndarray, shape (n_trials,) — valid event count per trial.

    CTR per trial = rewards / valid_events.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_offline_trial)(bandit_factory, env_factory)
        for _ in range(n_trials)
    )
    rewards = np.array([r for r, _ in results], dtype=float)
    valid_events = np.array([v for _, v in results], dtype=float)
    return rewards, valid_events
