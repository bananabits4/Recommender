"""
Collaborative Filtering Recommender
-------------------------------------
Builds a user-item matrix from ratings.csv and applies Truncated SVD to
learn latent factors. Predicts scores for all movies for a given user.

This is the collaborative leg of the hybrid pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, movies_path: str, ratings_path: str, n_components: int = 50):
        """
        Load ratings, build the user-item matrix, and fit SVD.

        Args:
            movies_path:  Path to movies.csv  (columns: movieId, title, genres)
            ratings_path: Path to ratings.csv (columns: userId, movieId, rating, timestamp)
            n_components: Number of latent factors for SVD (default 50)
        """
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        # --- Build user-item matrix ---
        # Rows = users, columns = movies, values = ratings (0 if not rated)
        self._matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )
        self._user_ids = self._matrix.index.tolist()
        self._movie_ids = self._matrix.columns.tolist()

        # --- Fit Truncated SVD ---
        n_components = min(n_components, min(self._matrix.shape) - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._user_factors = self._svd.fit_transform(self._matrix.values)
        # item_factors shape: (n_components, n_movies) → transpose to (n_movies, n_components)
        self._item_factors = self._svd.components_.T

        print(
            f"[Collaborative] {len(self._user_ids)} users, "
            f"{len(self._movie_ids)} movies, "
            f"{n_components} latent factors."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, n: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        """
        Return top-N movie recommendations for a given user.

        Args:
            user_id:      The user to generate recommendations for.
            n:            Number of results to return.
            exclude_seen: If True, skip movies the user has already rated.

        Returns a DataFrame with columns: movieId, title, genres, cf_score
        """
        if user_id not in self._user_ids:
            raise ValueError(f"User ID {user_id} not found in ratings data.")

        user_idx = self._user_ids.index(user_id)
        user_vec = self._user_factors[user_idx]  # shape: (n_components,)

        # Predicted scores for all movies: dot product of user vec × item factors
        scores = self._item_factors.dot(user_vec)  # shape: (n_movies,)

        scores_series = pd.Series(scores, index=self._movie_ids)

        # Optionally remove movies this user has already rated
        if exclude_seen:
            seen = self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
            scores_series = scores_series.drop(labels=seen, errors="ignore")

        top_ids = scores_series.nlargest(n).index.tolist()
        top_scores = scores_series[top_ids].tolist()

        result = self.movies[self.movies["movieId"].isin(top_ids)].copy()
        score_map = dict(zip(top_ids, top_scores))
        result["cf_score"] = result["movieId"].map(score_map)
        result = result.sort_values("cf_score", ascending=False).reset_index(drop=True)

        return result[["movieId", "title", "genres", "cf_score"]]

    def scores_for_user(self, user_id: int) -> dict:
        """
        Return a {movieId: cf_score} dict for every movie in the catalogue.
        Used by the fusion layer to blend with content-based scores.

        Returns an empty dict if the user is unknown (cold-start).
        """
        if user_id not in self._user_ids:
            return {}

        user_idx = self._user_ids.index(user_id)
        user_vec = self._user_factors[user_idx]
        scores = self._item_factors.dot(user_vec)

        return dict(zip(self._movie_ids, scores.tolist()))

    def is_known_user(self, user_id: int) -> bool:
        """Return True if this user exists in the training ratings."""
        return user_id in self._user_ids
