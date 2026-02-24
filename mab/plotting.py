"""Matplotlib helpers for reproducing paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_regret_vs_delta(
    results: dict[str, list[float]],  # label -> list of mean regrets (one per delta)
    deltas: list[float],
    title: str = "Flipping environment",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot mean regret vs delta (reproduces Fig 2a style).

    Parameters
    ----------
    results:
        Dict mapping algorithm label to list of mean regrets, one per delta value.
    deltas:
        The delta values on the x-axis.
    """
    colors = {"CUSUM-UCB": "red", "PHT-UCB": "gray", "SW-UCB": "green", "D-UCB": "blue"}
    markers = {"CUSUM-UCB": "o", "PHT-UCB": "s", "SW-UCB": "^", "D-UCB": "D"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, regrets in results.items():
        ax.plot(
            deltas,
            regrets,
            label=label,
            color=colors.get(label),
            marker=markers.get(label),
            markersize=4,
        )

    ax.set_xlabel("Parameter Δ")
    ax.set_ylabel("Regret")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_regret_vs_T(
    results: dict[str, list[float]],  # label -> list of mean regrets (one per T)
    T_values: list[int],
    title: str = "Switching environment",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot mean regret vs horizon T (reproduces Fig 2b style)."""
    colors = {"CUSUM-UCB": "red", "PHT-UCB": "gray", "SW-UCB": "green", "D-UCB": "blue"}
    markers = {"CUSUM-UCB": "o", "PHT-UCB": "s", "SW-UCB": "^", "D-UCB": "D"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, regrets in results.items():
        ax.plot(
            T_values,
            regrets,
            label=label,
            color=colors.get(label),
            marker=markers.get(label),
            markersize=4,
        )

    ax.set_xlabel("T")
    ax.set_ylabel("Regret")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
