"""MIND-small dataset loader.

Loads the MIND-small behaviors.tsv as a logged bandit dataset for offline
rejection-sampling evaluation.

Each impression row (one user session) shows several news articles simultaneously.
We expand impressions into individual (article_id, click) events in impression
order, which is approximately chronological.  This gives natural non-stationarity:
news articles peak in popularity as they break and fade as they age.

Arms = top-K most frequently shown articles across the entire split.

Dataset
-------
MIND-small train split (~65 MB):
  https://www.kaggle.com/datasets/arashnic/mind-news-dataset

Expected layout after extraction:
  data/mind/train/MINDsmall_train/behaviors.tsv

Reference
---------
Wu, F., Qiao, Y., Chen, J. H., et al. (2020).
MIND: A Large-scale Dataset for News Recommendation. ACL 2020.
"""

from pathlib import Path

import numpy as np


def load_mind(
    data_path: str | Path,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load MIND behaviors.tsv as a logged bandit dataset.

    Parameters
    ----------
    data_path : str or Path
        Directory containing behaviors.tsv.  The loader also searches
        sub-directories, so passing the parent of MINDsmall_train/ works too.
    k : int
        Number of arms (top-k most frequently shown articles).

    Returns
    -------
    logged_arms : ndarray, shape (T,)
        Arm index (0..k-1) for each logged event.
    logged_rewards : ndarray, shape (T,)
        Click label (1 = click, 0 = no-click).
    article_ids : list of str
        Mapping arm index → original MIND news ID (e.g. 'N12345').
    """
    path = Path(data_path)
    behaviors_path = path / "behaviors.tsv"
    if not behaviors_path.exists():
        candidates = sorted(path.glob("**/behaviors.tsv"))
        if not candidates:
            raise FileNotFoundError(
                f"behaviors.tsv not found under {data_path}.\n"
                "Expected: data/mind/train/MINDsmall_train/behaviors.tsv\n"
                "Download MIND-small from: "
                "https://www.kaggle.com/datasets/arashnic/mind-news-dataset"
            )
        behaviors_path = candidates[0]

    print(f"MIND: loading {behaviors_path}…")

    # --- Pass 1: count display frequencies to select top-K arms ---
    article_counts: dict[str, int] = {}
    with open(behaviors_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or not parts[4]:
                continue
            for imp in parts[4].split():
                aid = imp.split("-")[0]
                article_counts[aid] = article_counts.get(aid, 0) + 1

    top_k = sorted(article_counts, key=article_counts.__getitem__, reverse=True)[:k]
    article_set = set(top_k)
    id_to_arm = {aid: i for i, aid in enumerate(top_k)}

    # --- Pass 2: collect events in impression order (≈ chronological) ---
    # Each impression is expanded: one event per article in the top-K that appears
    # in the impression's candidate list.
    logged_arms: list[int] = []
    logged_rewards: list[float] = []

    with open(behaviors_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or not parts[4]:
                continue
            for imp in parts[4].split():
                aid, click = imp.split("-")
                if aid in article_set:
                    logged_arms.append(id_to_arm[aid])
                    logged_rewards.append(float(click))

    return (
        np.array(logged_arms, dtype=int),
        np.array(logged_rewards, dtype=float),
        top_k,
    )
