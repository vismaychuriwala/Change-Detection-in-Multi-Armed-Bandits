"""Yahoo! R6A dataset loader.

The R6A dataset requires **free registration** at Yahoo! Webscope:
  https://webscope.sandbox.yahoo.com/catalog.php?datatype=r

Select: "R6 - Yahoo! Front Page Today Module User Click Log (Version 1.0)"

After downloading and extracting, the dataset directory contains files named:
  ydata-fp-td-clicks-v1_0.XXXXXX

Each line format (space / pipe delimited):
  timestamp  article_id  click  |user  feat:val …  |article_id  feat:val …  …
                                                     ^-- candidate article pool --^

Arms are the top-K most-frequently displayed articles across the whole log.
Events where the displayed article is not in the top-K are discarded during
loading (they are not useful for rejection-sampling evaluation of K-arm bandits).

Reference
---------
Liu, H., Dolan, E., Zhou, H., & Shroff, N. (2018).  A Change-Detection based
Framework for Piecewise-stationary Multi-Armed Bandit Problem.  IEEE TNNLS.

Li, L. et al. (2010).  A Contextual-Bandit Approach to Personalized News
Article Recommendation.  WWW.
"""

from pathlib import Path

import numpy as np


def load_yahoo_r6a(
    data_path: str | Path,
    k: int = 10,
    max_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load Yahoo! R6A as a logged bandit dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to the R6A data **directory** (containing ydata-fp-td-clicks-v1_0.*
        files) or to a single such file.
    k : int
        Number of arms (top-k most frequently displayed articles).
    max_events : int, optional
        Cap on log lines to read.  Useful for quick smoke-testing; the full
        dataset contains ~36 M events.

    Returns
    -------
    logged_arms : ndarray, shape (T,)
        Arm index (0..k-1) for each retained log event.
    logged_rewards : ndarray, shape (T,)
        Click label (1 = click, 0 = no-click).
    article_ids : list of int
        Mapping arm index → original Yahoo article ID.

    Raises
    ------
    FileNotFoundError
        If no R6A data files are found at data_path.
    """
    path = Path(data_path)
    if path.is_dir():
        files = sorted(path.glob("ydata-fp-td-clicks-v1_0.*"))
    else:
        files = [path] if path.exists() else []

    if not files:
        raise FileNotFoundError(
            f"No R6A data files found at: {data_path}\n\n"
            "To obtain the Yahoo! R6A dataset:\n"
            "  1. Register (free) at https://webscope.sandbox.yahoo.com/\n"
            "  2. Request: 'R6 - Yahoo! Front Page Today Module User Click Log'\n"
            "  3. Extract the archive to a local directory\n"
            "  4. Pass that directory path to load_yahoo_r6a(data_path=…)\n"
        )

    # --- Pass 1: count how often each article was displayed ---
    print("Yahoo! R6A: scanning article display frequencies…")
    article_counts: dict[int, int] = {}
    lines_read = 0
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                if max_events is not None and lines_read >= max_events:
                    break
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                try:
                    article_id = int(tokens[1])
                except ValueError:
                    continue
                article_counts[article_id] = article_counts.get(article_id, 0) + 1
                lines_read += 1

    top_k_articles = sorted(article_counts, key=article_counts.__getitem__, reverse=True)[:k]
    article_set = set(top_k_articles)
    id_to_arm = {aid: i for i, aid in enumerate(top_k_articles)}

    # --- Pass 2: collect events whose displayed article is in the top-K ---
    print(f"Yahoo! R6A: loading events for top {k} articles…")
    logged_arms: list[int] = []
    logged_rewards: list[float] = []

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                try:
                    article_id = int(tokens[1])
                    click = float(tokens[2])
                except ValueError:
                    continue
                if article_id not in article_set:
                    continue
                logged_arms.append(id_to_arm[article_id])
                logged_rewards.append(click)

    return (
        np.array(logged_arms, dtype=int),
        np.array(logged_rewards, dtype=float),
        top_k_articles,
    )
