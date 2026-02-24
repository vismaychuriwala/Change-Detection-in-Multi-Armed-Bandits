"""Offline MAB benchmark on MIND-small news click dataset.

Arms   = top-K most frequently shown news articles.
Events = individual (article, click) pairs expanded from impression logs,
         in impression order (approximately chronological).
Reward = 1 if the user clicked, 0 otherwise.

News article click-through rates shift rapidly as stories break and fade,
creating natural piecewise-stationary non-stationarity for CD algorithms.

Evaluation uses rejection sampling (Li et al. 2010): at each timestep the
bandit proposes an article arm; the event counts only if that matches the
logged article.  Metric: CTR = total reward / valid events.

Dataset
-------
Download MIND-small from https://www.kaggle.com/datasets/arashnic/mind-news-dataset
Extract so that behaviors.tsv is at:  data/mind/train/MINDsmall_train/behaviors.tsv

Usage
-----
  python scripts/run_mind.py --k 10 --n-trials 10 --out assets/mind_ctr.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mab.bandits import CDUCB, DUCB, SWUCB
from mab.datasets.mind import load_mind
from mab.detectors import CUSUM, PHT
from mab.environments.replay import LoggedEnv
from mab.experiment import run_offline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/mind/train/MINDsmall_train",
        help="Path to directory containing behaviors.tsv",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of article arms (default 10)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", default="assets/mind_ctr.png")
    args = parser.parse_args()

    K = args.k

    logged_arms, logged_rewards, article_ids = load_mind(args.data, k=K)
    T = len(logged_arms)
    base_ctr = float(logged_rewards.mean())
    print(f"  {T:,} events  |  overall CTR (top-{K}) = {base_ctr:.4f}\n")

    def env_factory() -> LoggedEnv:
        return LoggedEnv(n_arms=K, logged_arms=logged_arms, logged_rewards=logged_rewards)

    # --- Hyper-parameters ---
    xi = 1.0      # UCB coefficient (ξ=1 per paper theorems)
    eps = 0.1     # change-detection sensitivity
    alpha = 0.05  # forced-exploration probability (small for large K)
    h_pht = 3.0
    h_cusum = 5.0
    M_cusum = 100
    M_sw = max(100, T // 500)   # ~0.2% of events
    gamma_ducb = 0.999

    configs: dict = {
        "CUSUM-UCB": lambda: CDUCB(
            k=K, xi=xi, alpha=alpha,
            detector_cls=CUSUM,
            detector_kwargs={"eps": eps, "M": M_cusum, "h": h_cusum},
        ),
        "PHT-UCB": lambda: CDUCB(
            k=K, xi=xi, alpha=alpha,
            detector_cls=PHT,
            detector_kwargs={"eps": eps, "h": h_pht},
        ),
        "SW-UCB": lambda: SWUCB(k=K, eps=xi, M=M_sw, alpha=alpha),
        "D-UCB": lambda: DUCB(k=K, gamma=gamma_ducb, eps=xi),
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
            bar.get_x() + bar.get_width() / 2, mean + std + 0.001,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean CTR")
    ax.set_title(f"MIND-small — Offline MAB Evaluation  (K={K} articles)")
    ax.legend()
    ax.set_ylim(0, max(means) * 1.4)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
