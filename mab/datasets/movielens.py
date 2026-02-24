"""MovieLens 100K dataset loader."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Genre names in the order they appear as binary columns in u.item (columns 5..23).
_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _ensure_downloaded(data_dir: Path) -> Path:
    """Download and extract ml-100k to data_dir/ml-100k/ if not already there."""
    dest = data_dir / "ml-100k"
    if (dest / "u.data").exists():
        return dest
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from GroupLens…")
    urlretrieve(_URL, zip_path)
    print("Extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    # GroupLens extracts to a subdirectory called ml-100k/
    extracted = data_dir / "ml-100k"
    if not (extracted / "u.data").exists():
        raise RuntimeError(f"Extraction failed: expected {extracted / 'u.data'}")
    return extracted


def load_movielens(
    k: int = 5,
    threshold: int = 4,
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load MovieLens 100K as a logged bandit dataset with genre arms.

    Parameters
    ----------
    k : int
        Number of arms (top-k genres by total rating count, excluding 'unknown').
    threshold : int
        Reward threshold: reward = 1 if rating >= threshold, else 0.
    data_dir : path, optional
        Directory to store/find the ml-100k download.
        Defaults to <project_root>/data/.

    Returns
    -------
    logged_arms : ndarray, shape (T,)
        Arm index (0..k-1) for each logged event.
    logged_rewards : ndarray, shape (T,)
        Binary reward (1 = liked, 0 = disliked) for each logged event.
    arm_names : list of str
        Genre name for each arm index.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir = Path(data_dir)
    path = _ensure_downloaded(data_dir)

    # --- Load ratings: user_id, item_id, rating, timestamp ---
    ratings = np.loadtxt(path / "u.data", dtype=int)

    # --- Load movie genre flags ---
    # u.item columns: item_id | title | release | video_release | imdb | g0 | g1 | ... | g18
    item_genre: dict[int, int] = {}  # item_id -> genre index into _GENRES
    with open(path / "u.item", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            item_id = int(parts[0])
            genre_flags = [int(x) for x in parts[5:24]]  # exactly 19 genre columns
            # Primary genre = first flagged bit, preferring non-unknown (index > 0).
            primary = next(
                (i for i, g in enumerate(genre_flags) if g and i > 0),
                next((i for i, g in enumerate(genre_flags) if g), 0),
            )
            item_genre[item_id] = primary

    # --- Count ratings per genre (excluding 'unknown') ---
    genre_counts: dict[int, int] = {}
    for row in ratings:
        g = item_genre.get(int(row[1]), 0)
        if g > 0:  # skip 'unknown'
            genre_counts[g] = genre_counts.get(g, 0) + 1

    if len(genre_counts) < k:
        raise ValueError(
            f"Only {len(genre_counts)} non-unknown genres found; requested k={k}."
        )

    top_genres = sorted(genre_counts, key=genre_counts.__getitem__, reverse=True)[:k]
    genre_to_arm = {g: i for i, g in enumerate(top_genres)}
    arm_names = [_GENRES[g] for g in top_genres]

    # --- Build logged dataset: filter to top-k genres, sort by timestamp ---
    events: list[tuple[int, int, int]] = []  # (rating, timestamp, genre_idx)
    for row in ratings:
        g = item_genre.get(int(row[1]), 0)
        if g in genre_to_arm:
            events.append((int(row[2]), int(row[3]), genre_to_arm[g]))

    events.sort(key=lambda x: x[1])  # ascending timestamp

    logged_rewards = np.array([1 if r >= threshold else 0 for r, _, _ in events], dtype=float)
    logged_arms = np.array([arm for _, _, arm in events], dtype=int)

    return logged_arms, logged_rewards, arm_names
