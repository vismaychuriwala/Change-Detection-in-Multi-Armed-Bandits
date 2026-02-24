"""cd-mab: Change-Detection Multi-Armed Bandit algorithms in Python.

Quick start
-----------
>>> from mab.detectors import CUSUM, PHT
>>> from mab.bandits import CDUCB, SWUCB, DUCB
>>> from mab.environments import FlippingEnv, SwitchingEnv, LoggedEnv
>>> from mab.experiment import run_experiment, run_offline_experiment
>>> from mab.datasets import load_movielens, load_yahoo_r6a
"""

from .detectors import CUSUM, PHT, ChangeDetector
from .bandits import CDUCB, SWUCB, DUCB, BanditAlgorithm
from .environments import FlippingEnv, SwitchingEnv, Environment, LoggedEnv
from .experiment import run_experiment, run_trial, run_offline_experiment, run_offline_trial

__all__ = [
    "ChangeDetector", "CUSUM", "PHT",
    "BanditAlgorithm", "CDUCB", "SWUCB", "DUCB",
    "Environment", "FlippingEnv", "SwitchingEnv", "LoggedEnv",
    "run_experiment", "run_trial", "run_offline_experiment", "run_offline_trial",
]
