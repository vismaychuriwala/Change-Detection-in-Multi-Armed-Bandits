"""Switching environment experiment: regret vs horizon."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mab.bandits import CDUCB, SWUCB, DUCB
from mab.detectors import CUSUM, PHT
from mab.environments import SwitchingEnv
from mab.experiment import run_experiment
from mab.plotting import plot_regret_vs_T


def main() -> None:
    parser = argparse.ArgumentParser(description="Switching environment experiment")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="switching_regret.png")
    args = parser.parse_args()

    K = 5
    beta = 0.2
    T_values = list(range(1000, 10_001, 1000))

    h = np.log(1 / beta)
    alpha = np.sqrt(beta * np.log(1 / beta))
    xi = 1.0   # UCB exploration coefficient (ξ=1, per Theorems 1 and 3)
    eps = 0.1  # change-detection sensitivity (ε in CUSUM/PHT)
    gamma_ducb = 1.0 - beta  # discount factor for D-UCB

    print(f"Switching env | K={K}, beta={beta}, trials={args.n_trials}")
    print(f"  alpha={alpha:.4f}, h={h:.4f}, xi={xi}, eps={eps}, gamma(D-UCB)={gamma_ducb}")
    print(f"  T values: {T_values}\n")

    results: dict[str, list[float]] = {
        "CUSUM-UCB": [], "PHT-UCB": [], "SW-UCB": [], "D-UCB": []
    }

    for T in T_values:
        print(f"  T={T}", end="  ", flush=True)

        # CUSUM-UCB (Algorithm 3: warmup=M so each arm gets M forced plays for û₀ estimation)
        regrets = run_experiment(
            bandit_factory=lambda: CDUCB(
                k=K, xi=xi, alpha=alpha,
                detector_cls=CUSUM,
                detector_kwargs={"eps": eps, "M": 100, "h": h},
                warmup=100,
            ),
            env_factory=lambda t=T: SwitchingEnv(_T=t, k=K, beta=beta),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["CUSUM-UCB"].append(float(regrets.mean()))
        print(f"CUSUM={regrets.mean():.1f}", end="  ", flush=True)

        # PHT-UCB (no warmup needed — PHT uses running mean, no fixed burn-in)
        regrets = run_experiment(
            bandit_factory=lambda: CDUCB(
                k=K, xi=xi, alpha=alpha,
                detector_cls=PHT,
                detector_kwargs={"eps": eps, "h": h},
            ),
            env_factory=lambda t=T: SwitchingEnv(_T=t, k=K, beta=beta),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["PHT-UCB"].append(float(regrets.mean()))
        print(f"PHT={regrets.mean():.1f}", end="  ", flush=True)

        # SW-UCB
        regrets = run_experiment(
            bandit_factory=lambda: SWUCB(k=K, eps=eps, M=30, alpha=alpha),
            env_factory=lambda t=T: SwitchingEnv(_T=t, k=K, beta=beta),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["SW-UCB"].append(float(regrets.mean()))
        print(f"SW={regrets.mean():.1f}", end="  ", flush=True)

        # D-UCB (extension)
        regrets = run_experiment(
            bandit_factory=lambda: DUCB(k=K, gamma=gamma_ducb, eps=eps),
            env_factory=lambda t=T: SwitchingEnv(_T=t, k=K, beta=beta),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["D-UCB"].append(float(regrets.mean()))
        print(f"D-UCB={regrets.mean():.1f}")

    fig = plot_regret_vs_T(results, T_values, save_path=args.out)
    print(f"\nPlot saved to {args.out}")
    fig.show()


if __name__ == "__main__":
    main()
