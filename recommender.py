"""
Content-Based Movie Recommender
--------------------------------
Uses TF-IDF on movie genres + titles to find similar movies via cosine similarity.
This is the core ML logic — no UI, no API. Just the recommendation engine.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movies_path: str, ratings_path: str = None):
        """
        Load movies (and optionally ratings) and build the similarity matrix.

        Args:
            movies_path: Path to movies.csv  (columns: movieId, title, genres)
            ratings_path: Path to ratings.csv (columns: userId, movieId, rating, timestamp)
                          Optional — only needed for collaborative filtering later.
        """
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path) if ratings_path else None

        self.movies["genres_clean"] = self.movies["genres"].str.replace("|", " ", regex=False)

        self.movies["soup"] = (
            self.movies["title"].str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
            + " "
            + self.movies["genres_clean"]
        )

        self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(self.movies["soup"])

        self._sim_matrix = cosine_similarity(self._tfidf_matrix, self._tfidf_matrix)

        self._title_to_idx = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )

        print(f"[Recommender] Loaded {len(self.movies)} movies.")
        if self.ratings is not None:
            print(f"[Recommender] Loaded {len(self.ratings)} ratings from {self.ratings['userId'].nunique()} users.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:

        idx = self._find_movie_index(title)
        sim_scores = list(enumerate(self._sim_matrix[idx]))

        # Sort by similarity score descending, skip index 0 (the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, score) for i, score in sim_scores if i != idx][:n]

        movie_indices = [i for i, _ in sim_scores]
        scores = [round(score, 4) for _, score in sim_scores]

        result = self.movies.iloc[movie_indices][["title", "genres"]].copy()
        result["similarity_score"] = scores
        result = result.reset_index(drop=True)
        result.index += 1  # 1-based rank for display

        return result

    def search(self, query: str) -> pd.DataFrame:
        
        mask = self.movies["title"].str.lower().str.contains(query.lower(), na=False)
        return self.movies[mask][["movieId", "title", "genres"]].reset_index(drop=True)

    def movie_info(self, title: str) -> dict:
        """Return metadata for a single movie by title."""
        idx = self._find_movie_index(title)
        row = self.movies.iloc[idx]
        return {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_movie_index(self, title: str) -> int:
        """
        Resolve a movie title to its DataFrame row index.
        Tries exact match first, then falls back to partial/fuzzy match.
        """
        title_lower = title.lower().strip()

        # 1. Exact match
        if title_lower in self._title_to_idx:
            return int(self._title_to_idx[title_lower])

        # 2. Partial match — pick the first result
        matches = self.movies[
            self.movies["title"].str.lower().str.contains(title_lower, na=False)
        ]
        if not matches.empty:
            matched_title = matches.iloc[0]["title"]
            print(f"[Recommender] Partial match: '{title}' → '{matched_title}'")
            return int(matches.index[0])

        raise ValueError(
            f"Movie '{title}' not found. Use recommender.search('{title}') to browse options."
        )
