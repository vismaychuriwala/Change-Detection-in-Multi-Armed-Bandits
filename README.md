# Change-Detection Bandits

Standard UCB assumes the reward distributions are stationary. In the real world they're not. A news article's click-through rate drops once the story goes stale. A product stops being the best recommendation after a trend shifts. This project is about what happens when you add change detectors to UCB algorithms — how well it works, and when it doesn't.

I originally implemented this in Wolfram Mathematica as a course project at IIT Madras, replicating experiments from Liu et al. (2018). This is the Python port, extended with a third bandit algorithm (D-UCB) and validated on real click-through data from Microsoft's MIND news dataset.

---

## The problem

A multi-armed bandit has K arms with unknown reward distributions. Standard UCB deals well with stationary rewards — it pulls an arm more confidently the more data it has that the arm is good. Non-stationarity breaks this: the best arm today might be the worst arm tomorrow, but UCB keeps exploiting stale historical data.

Two strategies for handling it:

**Active:** Attach a change detector to each arm. When it fires, treat that arm as unknown and start fresh — reset its statistics, leave the others alone.

**Passive:** Forget old data deliberately. Keep only the last M observations (sliding window), or weight older ones less (geometric discounting). No explicit detection, just structured forgetting.

---

## Algorithms

### Change detectors

Both detectors work by watching whether an arm's reward distribution has shifted from a reference level.

**CUSUM** fixes the baseline from the first M observations. After burn-in it runs cumulative sum statistics and fires when either exceeds threshold h:

```
g⁺ₜ = max(0, g⁺ₜ₋₁ + (xₜ − μ₀ − ε))
g⁻ₜ = max(0, g⁻ₜ₋₁ + (μ₀ − xₜ − ε))
alarm if g⁺ₜ ≥ h or g⁻ₜ ≥ h
```

**PHT** (Page-Hinkley Test) is the same walk, but uses the running mean of all observations as the baseline rather than a fixed M-sample estimate. No burn-in required.

On alarm, that arm's history resets. Detectors are arm-independent — a change in arm 2 doesn't touch the state of arms 1, 3, 4, 5.

### Bandit algorithms

| Algorithm | Type | How it handles change |
|---|---|---|
| CUSUM-UCB | Active | CUSUM detector per arm; resets arm on alarm |
| PHT-UCB | Active | PHT detector per arm; resets arm on alarm |
| SW-UCB | Passive | Sliding window of last M rewards per arm |
| D-UCB | Passive | Geometric discounting — older rewards weighted by γᵗ⁻ˢ |

All use UCB arm selection with exploration coefficient ξ = 1 (required for the regret bound). The active algorithms add a small forced-exploration probability α so they don't miss changes in arms they've stopped pulling.

---

## Synthetic experiments

### Flipping environment

Two arms. Arm 0 is always Bernoulli(0.5). Arm 1 cycles between 0.5 − Δ and 0.8 at deterministic changepoints T/3 and 2T/3. Small Δ = hard (arms are similar); large Δ = easy (arm 1 is clearly better or worse).

![Flipping environment: regret vs Δ](assets/flipping_regret.png)

Both active methods outperform SW-UCB, and the gap grows with Δ. When the arms are easy to distinguish the detector fires quickly after a change and resets cleanly. At small Δ the signal is weak — it takes many samples to build up the walk statistic — so detection is slow and the advantage shrinks.

### Switching environment

Five arms. At each step, every arm independently redraws its mean from U[0,1] with probability β = 0.2. This is a hazard-function model: changepoints arrive continuously and randomly per arm.

![Switching environment: regret vs T](assets/switching_regret.png)

Completely different story. D-UCB has roughly 25% lower regret than everything else. CUSUM and PHT barely improve over SW-UCB.

The reason is calibration. The switching model's hazard rate β directly determines the optimal discount factor: γ = 1 − β = 0.8, giving D-UCB an effective memory of 1/(1 − γ) = 5 steps. That matches the expected 1/β = 5 steps between changepoints exactly. CUSUM needs M = 100 samples just for burn-in — far longer than the typical time between changes. By the time a detector accumulates enough evidence to fire, the distribution has already changed several more times. Smooth exponential forgetting with the right γ is simply a better fit for very high-frequency change.

---

## Real data: MIND news clicks

I ran the same algorithms on Microsoft's MIND-small dataset (156k impression logs, ~5.8M article display events). News click-through rates have natural non-stationarity: a story breaks, peaks quickly, then fades as the news cycle moves on.

For evaluation I used rejection sampling (Li et al. 2010): at each step the bandit proposes an article arm; the event counts only if that matches the article that was actually displayed in the log. This gives an unbiased estimate of the algorithm's CTR without needing a live deployment.

Arms are the 10 most frequently displayed articles, covering ~181k events. The overall CTR for these articles is 7.9%.

![MIND-small: CTR over time](assets/mind_learning_curve.png)

All three UCB variants converge quickly and plateau around 20–22% CTR — roughly 2.8× the random baseline. The learning curve shows this happens within the first ~2000 valid events.

![MIND-small: CTR by algorithm](assets/mind_ctr.png)

D-UCB is the interesting case. It won the switching environment but underperforms here at 12.2%. Two reasons. First, γ = 0.999 is too slow — it gives an effective memory of ~1000 steps, which is far too inert for articles that peak and fade within a few hundred impressions. Second, D-UCB has no forced exploration (no α term), so once it commits to an article it rarely re-explores, and misses articles that have since become more popular. In the synthetic switching env this didn't matter because γ was tuned to match β exactly. Here there's no such prior.

PHT-UCB edges CUSUM-UCB slightly because news article CTR shifts tend to be abrupt (sudden break → fast fade), which PHT detects without needing a fixed burn-in period. CUSUM's M = 100 sample burn-in takes longer to establish the pre-change baseline.

---

## Reference

Liu, H., Dolan, E., Zhou, H., & Shroff, N. (2018). A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem. *IEEE Transactions on Neural Networks and Learning Systems*.
