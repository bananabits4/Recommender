"""
Hybrid Fusion Recommender
--------------------------
Blends Content-Based (CB) and Collaborative Filtering (CF) scores
using a weighted alpha parameter:

    final_score = alpha * CF_score + (1 - alpha) * CB_score

Cold-start handling:
    - Unknown user_id → alpha forced to 0.0 (pure content-based)
    - No title provided → alpha forced to 1.0 (pure collaborative)
"""

import pandas as pd

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(
        self,
        movies_path: str,
        ratings_path: str,
        n_components: int = 50,
    ):
        """
        Initialise and fit both sub-models.

        Args:
            movies_path:   Path to movies.csv
            ratings_path:  Path to ratings.csv
            n_components:  Latent factors for SVD in the CF model
        """
        self.cb = ContentBasedRecommender(movies_path=movies_path)
        self.cf = CollaborativeRecommender(
            movies_path=movies_path,
            ratings_path=ratings_path,
            n_components=n_components,
        )
        # Expose movies df directly for the API layer
        self.movies = self.cb.movies

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        title: str | None = None,
        user_id: int | None = None,
        n: int = 5,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """
        Return top-N hybrid recommendations.

        Args:
            title:   Movie title for the content-based signal.
            user_id: User ID for the collaborative signal.
            n:       Number of results.
            alpha:   CF weight in [0, 1]. If None, it is chosen automatically:
                     - Known user + title  → 0.5 (equal blend)
                     - Known user, no title → 1.0 (pure CF)
                     - Unknown user / no user_id → 0.0 (pure CB)

        Returns a DataFrame with: movieId, title, genres, cb_score, cf_score, final_score
        """
        known_user = user_id is not None and self.cf.is_known_user(user_id)
        has_title = title is not None and title.strip() != ""

        # --- Determine alpha automatically if not supplied ---
        if alpha is None:
            if known_user and has_title:
                alpha = 0.5
            elif known_user and not has_title:
                alpha = 1.0
            else:
                alpha = 0.0  # cold-start: fall back to content-based

        # --- Gather scores from each model ---
        cb_scores: dict = {}
        if has_title:
            cb_scores = self.cb.scores_for_all(title)

        cf_scores: dict = {}
        if known_user:
            cf_scores = self.cf.scores_for_user(user_id)

        # --- Normalise each score dict to [0, 1] ---
        cb_scores = _min_max_normalise(cb_scores)
        cf_scores = _min_max_normalise(cf_scores)

        # --- Blend ---
        all_movie_ids = set(cb_scores) | set(cf_scores)
        rows = []
        for mid in all_movie_ids:
            cb = cb_scores.get(mid, 0.0)
            cf = cf_scores.get(mid, 0.0)
            final = alpha * cf + (1 - alpha) * cb
            rows.append({"movieId": mid, "cb_score": cb, "cf_score": cf, "final_score": final})

        if not rows:
            return pd.DataFrame(columns=["movieId", "title", "genres", "cb_score", "cf_score", "final_score"])

        scores_df = pd.DataFrame(rows)

        # --- Exclude the seed movie itself ---
        if has_title:
            try:
                seed_info = self.cb.movie_info(title)
                scores_df = scores_df[scores_df["movieId"] != seed_info["movieId"]]
            except ValueError:
                pass

        # --- Pick top-N and join movie metadata ---
        top = scores_df.nlargest(n, "final_score")
        result = top.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        result = result[["movieId", "title", "genres", "cb_score", "cf_score", "final_score"]]
        result = result.reset_index(drop=True)
        result.index += 1  # 1-based rank

        # Round display scores
        for col in ("cb_score", "cf_score", "final_score"):
            result[col] = result[col].round(4)

        return result

    def search(self, query: str) -> pd.DataFrame:
        """Delegate title search to the content-based model."""
        return self.cb.search(query)

    def movie_info(self, title: str) -> dict:
        """Delegate single-movie lookup to the content-based model."""
        return self.cb.movie_info(title)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _min_max_normalise(scores: dict) -> dict:
    """Scale a {id: score} dict so values lie in [0, 1]."""
    if not scores:
        return scores
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
