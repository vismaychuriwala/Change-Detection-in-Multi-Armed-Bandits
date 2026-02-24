"""Reproduce the flipping-environment regret vs Δ experiment (Fig 2a).

Usage:
    python scripts/run_flipping.py [--n-trials N] [--n-jobs J] [--out PATH]

Parameters match the Mathematica reproduction:
    T = 999, K = 2, 20 trials, Δ ∈ {0.02, 0.04, ..., 0.30}
    CUSUM: ε=0.1, M=100, h=log(T/2), α=sqrt((2/T)*log(T/2))
    PHT  : ε=0.1,        h=log(T/2), α=sqrt((2/T)*log(T/2))
    SW   : ε=0.1, M=30,  h=log(T/2), α=sqrt((2/T)*log(T/2))
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mab.bandits import CDUCB, SWUCB
from mab.detectors import CUSUM, PHT
from mab.environments import FlippingEnv
from mab.experiment import run_experiment
from mab.plotting import plot_regret_vs_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Flipping environment experiment")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="flipping_regret.png")
    args = parser.parse_args()

    T = 999
    K = 2
    deltas = [round(d, 2) for d in np.arange(0.02, 0.31, 0.02)]

    log_half_T = np.log(T / 2)
    alpha = np.sqrt((2 / T) * log_half_T)
    h = log_half_T
    xi = 1.0   # UCB exploration coefficient (ξ=1, per Theorems 1 and 3)
    eps = 0.1  # change-detection sensitivity (ε in CUSUM/PHT)

    print(f"Flipping env | T={T}, K={K}, trials={args.n_trials}")
    print(f"  alpha={alpha:.4f}, h={h:.4f}, xi={xi}, eps={eps}")
    print(f"  deltas: {deltas}\n")

    results: dict[str, list[float]] = {"CUSUM-UCB": [], "PHT-UCB": [], "SW-UCB": []}

    for delta in deltas:
        print(f"  delta={delta:.2f}", end="  ", flush=True)

        # CUSUM-UCB (Algorithm 3: warmup=M so each arm gets M forced plays for û₀ estimation)
        regrets = run_experiment(
            bandit_factory=lambda: CDUCB(
                k=K, xi=xi, alpha=alpha,
                detector_cls=CUSUM,
                detector_kwargs={"eps": eps, "M": 100, "h": h},
                warmup=100,
            ),
            env_factory=lambda d=delta: FlippingEnv(_T=T, delta=d),
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
            env_factory=lambda d=delta: FlippingEnv(_T=T, delta=d),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["PHT-UCB"].append(float(regrets.mean()))
        print(f"PHT={regrets.mean():.1f}", end="  ", flush=True)

        # SW-UCB
        regrets = run_experiment(
            bandit_factory=lambda: SWUCB(k=K, eps=eps, M=30, alpha=alpha),
            env_factory=lambda d=delta: FlippingEnv(_T=T, delta=d),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        results["SW-UCB"].append(float(regrets.mean()))
        print(f"SW={regrets.mean():.1f}")

    fig = plot_regret_vs_delta(results, deltas, save_path=args.out)
    print(f"\nPlot saved to {args.out}")
    fig.show()


if __name__ == "__main__":
    main()
