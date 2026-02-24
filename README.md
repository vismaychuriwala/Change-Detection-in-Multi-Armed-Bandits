# cd-mab

Python implementation of **Change Detection + UCB** algorithms for the Piecewise Stationary Multi-Armed Bandit problem.

Based on: *"A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem"* (Liu et al., IEEE TNNLS 2018).

Originally implemented in Wolfram Mathematica as part of a course project at IIT Madras.

---

## Algorithms

### Change Detectors (plug-in modules)
| Class | Algorithm | Key Idea |
|---|---|---|
| `CUSUM` | Two-sided CUSUM | Fixed baseline from first M samples; walks reset on alarm |
| `PHT` | Page-Hinkley Test | Running mean as adaptive baseline |

### Bandit Algorithms
| Class | Type | Description |
|---|---|---|
| `CDUCB` | Active (CD-UCB) | UCB + pluggable change detector; resets arm on alarm |
| `SWUCB` | Passive | Sliding window of last M rewards per arm |
| `DUCB` | Passive | Geometric discounting of older rewards (extension) |

---

## Installation

```bash
pip install -e ".[dev]"
```

Or just install dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from mab.detectors import CUSUM
from mab.bandits import CDUCB, SWUCB, DUCB
from mab.environments import FlippingEnv, SwitchingEnv
from mab.experiment import run_experiment
import numpy as np

T = 999
h = np.log(T / 2)
alpha = np.sqrt((2 / T) * h)

# CD-UCB with CUSUM change detector
regrets = run_experiment(
    bandit_factory=lambda: CDUCB(
        k=2, xi=1.0, alpha=alpha,
        detector_cls=CUSUM,
        detector_kwargs={"eps": 0.1, "M": 100, "h": h},
    ),
    env_factory=lambda: FlippingEnv(_T=T, delta=0.2),
    n_trials=20,
)
print(f"Mean regret: {regrets.mean():.2f} ± {regrets.std():.2f}")
```

---

## Reproducing Experiments

**Flipping environment** (regret vs Δ, K=2):
```bash
python scripts/run_flipping.py --n-trials 20 --out flipping_regret.png
```

**Switching environment** (regret vs T, K=5, + D-UCB extension):
```bash
python scripts/run_switching.py --n-trials 20 --out switching_regret.png
```

---

## Real-World Datasets (Offline Evaluation)

Both real-world datasets use **rejection-sampling** evaluation (Li et al. 2010):
at each timestep the bandit proposes an arm; the event is valid only if that
matches the arm that was actually logged.  Metric: **CTR = reward / valid events**.

### MovieLens 100K (auto-download)

Arms = top-K movie genres.  Events sorted by timestamp give gradual taste-shift
non-stationarity.

```bash
python scripts/run_movielens.py --k 5 --n-trials 20 --out movielens_ctr.png
```

The dataset (~5 MB) is downloaded automatically from GroupLens on first run.

### Yahoo! R6A Click Log (registration required)

Arms = top-K news articles.  News CTR shifts rapidly as stories break and fade,
making this the primary real-data benchmark from the paper.

```bash
# 1. Register (free) at https://webscope.sandbox.yahoo.com/
# 2. Download "R6 - Yahoo! Front Page Today Module User Click Log"
# 3. Extract to a directory, then:
python scripts/run_yahoo.py --data /path/to/r6a/ --k 10 --n-trials 10 --out yahoo_ctr.png

# Quick smoke test with first 200 000 events:
python scripts/run_yahoo.py --data /path/to/r6a/ --k 10 --max-events 200000 --n-trials 5
```

### Using any logged dataset

```python
import numpy as np
from mab.environments import LoggedEnv
from mab.bandits import CDUCB
from mab.detectors import PHT
from mab.experiment import run_offline_experiment

# logged_arms  : shape (T,) — arm index that was displayed at each step
# logged_rewards: shape (T,) — binary reward for the displayed arm
logged_arms = np.array([...])
logged_rewards = np.array([...])

rewards, valid = run_offline_experiment(
    bandit_factory=lambda: CDUCB(
        k=5, xi=1.0, alpha=0.1,
        detector_cls=PHT,
        detector_kwargs={"eps": 0.1, "h": 3.0},
    ),
    env_factory=lambda: LoggedEnv(n_arms=5, logged_arms=logged_arms, logged_rewards=logged_rewards),
    n_trials=10,
)
print(f"CTR: {(rewards / valid).mean():.4f}")
```

---

## Tests

```bash
pytest
```

---

## Structure

```
mab/
├── detectors/      # CUSUM, PHT
├── bandits/        # CDUCB, SWUCB, DUCB
├── environments/   # FlippingEnv, SwitchingEnv, LoggedEnv
├── datasets/       # load_movielens(), load_yahoo_r6a()
├── experiment.py   # run_experiment() + run_offline_experiment()
└── plotting.py     # matplotlib figure helpers
scripts/
├── run_flipping.py
├── run_switching.py
├── run_movielens.py
└── run_yahoo.py
tests/
data/               # downloaded datasets (gitignored)
```
