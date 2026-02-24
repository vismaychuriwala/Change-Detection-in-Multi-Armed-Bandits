"""Offline MAB benchmark on MovieLens 100K.

Arms   = top-K movie genres (by total rating count).
Events = all ratings for movies belonging to those genres, sorted by timestamp.
Reward = 1 if rating >= threshold (default 4), else 0.

Evaluation uses rejection sampling (Li et al. 2010): at each timestep the
bandit proposes a genre arm; the event counts only if that matches the genre
of the movie the user actually rated.  CTR = total reward / valid events.

The temporal ordering of ratings introduces gradual taste-shift
non-stationarity, which is what the CD algorithms are designed to exploit.

Usage
-----
  python scripts/run_movielens.py --k 5 --n-trials 10 --out movielens_ctr.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mab.bandits import CDUCB, DUCB, SWUCB
from mab.datasets import load_movielens
from mab.detectors import CUSUM, PHT
from mab.environments.replay import LoggedEnv
from mab.experiment import run_offline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Number of genre arms (default 5)")
    parser.add_argument("--threshold", type=int, default=4,
                        help="Rating threshold for positive reward (default 4)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", default="movielens_ctr.png")
    args = parser.parse_args()

    K = args.k

    print(f"Loading MovieLens 100K  (K={K} genre arms, threshold={args.threshold})…")
    logged_arms, logged_rewards, arm_names = load_movielens(k=K, threshold=args.threshold)
    T = len(logged_arms)
    base_ctr = float(logged_rewards.mean())
    print(f"  {T:,} events  |  arms: {arm_names}  |  overall CTR = {base_ctr:.4f}\n")

    def env_factory() -> LoggedEnv:
        return LoggedEnv(n_arms=K, logged_arms=logged_arms, logged_rewards=logged_rewards)

    # --- Shared hyper-parameters ---
    xi = 1.0      # UCB coefficient (ξ=1 per Theorems 1 and 3)
    eps = 0.1     # change-detection sensitivity
    alpha = 0.1   # forced-exploration probability
    h = 3.0       # alarm threshold for PHT / CUSUM
    M_sw = max(50, T // 200)   # SW-UCB window size (~0.5% of events)

    configs: dict[str, object] = {
        "CUSUM-UCB": lambda: CDUCB(
            k=K, xi=xi, alpha=alpha,
            detector_cls=CUSUM,
            detector_kwargs={"eps": eps, "M": 100, "h": h},
        ),
        "PHT-UCB": lambda: CDUCB(
            k=K, xi=xi, alpha=alpha,
            detector_cls=PHT,
            detector_kwargs={"eps": eps, "h": h},
        ),
        "SW-UCB": lambda: SWUCB(k=K, eps=xi, M=M_sw, alpha=alpha),
        "D-UCB": lambda: DUCB(k=K, gamma=0.995, eps=xi),
    }

    print(f"Running offline experiments ({args.n_trials} trials each)…\n")
    ctrs: dict[str, tuple[float, float]] = {}
    for name, factory in configs.items():
        rewards, valid = run_offline_experiment(
            bandit_factory=factory,
            env_factory=env_factory,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        ctr = rewards / np.maximum(valid, 1)
        ctrs[name] = (float(ctr.mean()), float(ctr.std()))
        print(
            f"  {name:<12}  CTR = {ctr.mean():.4f} ± {ctr.std():.4f}"
            f"   (valid events: {valid.mean():.0f} ± {valid.std():.0f})"
        )

    # --- Bar chart ---
    names = list(ctrs.keys())
    means = [ctrs[n][0] for n in names]
    stds = [ctrs[n][1] for n in names]
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55, color=colours)
    ax.axhline(base_ctr, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random CTR = {base_ctr:.3f}")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, mean + std + 0.003,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean CTR")
    ax.set_title(f"MovieLens 100K — Offline MAB Evaluation  (K={K} genres)")
    ax.legend()
    ax.set_ylim(0, max(means) * 1.35)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
