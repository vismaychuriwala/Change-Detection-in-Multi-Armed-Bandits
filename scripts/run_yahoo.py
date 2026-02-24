"""Offline MAB benchmark on Yahoo! R6A click log.

Arms   = top-K most frequently displayed news articles.
Events = all log entries where the displayed article is in the top-K.
Reward = 1 if the user clicked, 0 otherwise.

Evaluation uses rejection sampling (Li et al. 2010 / Liu et al. 2018):
at each step the bandit proposes an article arm; the event is only counted
if that matches the logged article.  CTR = total reward / valid events.

The dataset requires **free registration** at Yahoo! Webscope:
  https://webscope.sandbox.yahoo.com/catalog.php?datatype=r
  Dataset: "R6 - Yahoo! Front Page Today Module User Click Log (Version 1.0)"

Usage
-----
  # full dataset (slow first run — ~36 M events)
  python scripts/run_yahoo.py --data /path/to/r6a/ --k 10 --n-trials 10

  # quick smoke test with first 200 000 events
  python scripts/run_yahoo.py --data /path/to/r6a/ --k 10 --max-events 200000 --n-trials 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mab.bandits import CDUCB, DUCB, SWUCB
from mab.datasets.yahoo import load_yahoo_r6a
from mab.detectors import CUSUM, PHT
from mab.environments.replay import LoggedEnv
from mab.experiment import run_offline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="Path to the R6A directory or single data file.")
    parser.add_argument("--k", type=int, default=10, help="Number of article arms (default 10)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Cap on log lines to read (useful for quick testing)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", default="yahoo_ctr.png")
    args = parser.parse_args()

    K = args.k

    print(f"Loading Yahoo! R6A  (K={K} article arms)…")
    logged_arms, logged_rewards, article_ids = load_yahoo_r6a(
        args.data, k=K, max_events=args.max_events
    )
    T = len(logged_arms)
    base_ctr = float(logged_rewards.mean())
    print(f"  {T:,} events  |  overall CTR = {base_ctr:.4f}\n")

    def env_factory() -> LoggedEnv:
        return LoggedEnv(n_arms=K, logged_arms=logged_arms, logged_rewards=logged_rewards)

    # --- Shared hyper-parameters ---
    xi = 1.0
    eps = 0.1
    alpha = 0.05      # lower forced exploration for larger K
    h_pht = 3.0
    h_cusum = 5.0
    M_cusum = 200
    M_sw = max(100, T // 500)
    gamma_ducb = 0.999  # slow discounting for large T

    configs: dict[str, object] = {
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
    ax.set_title(f"Yahoo! R6A — Offline MAB Evaluation  (K={K} articles)")
    ax.legend()
    ax.set_ylim(0, max(means) * 1.35)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
