"""
Content-Based Movie Recommender
--------------------------------
Uses TF-IDF on movie genres + titles to find similar movies via cosine similarity.
This is the content-based leg of the hybrid pipeline.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_path: str):
        """
        Load movies and build the TF-IDF similarity matrix.

        Args:
            movies_path: Path to movies.csv (columns: movieId, title, genres)
        """
        self.movies = pd.read_csv(movies_path)

        # Build a text "soup" from title + genres for each movie
        self.movies["genres_clean"] = self.movies["genres"].str.replace("|", " ", regex=False)
        self.movies["soup"] = (
            self.movies["title"].str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
            + " "
            + self.movies["genres_clean"]
        )

        # Fit TF-IDF and compute full cosine similarity matrix
        self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(self.movies["soup"])
        self._sim_matrix = cosine_similarity(self._tfidf_matrix, self._tfidf_matrix)

        # Fast title → index lookup (lowercase)
        self._title_to_idx = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )

        print(f"[ContentBased] Loaded {len(self.movies)} movies.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        """
        Return top-N movies most similar to the given title.

        Returns a DataFrame with columns: title, genres, similarity_score
        """
        idx = self._find_movie_index(title)
        sim_scores = list(enumerate(self._sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, score) for i, score in sim_scores if i != idx][:n]

        movie_indices = [i for i, _ in sim_scores]
        scores = [round(score, 4) for _, score in sim_scores]

        result = self.movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
        result["cb_score"] = scores
        return result.reset_index(drop=True)

    def scores_for_all(self, title: str) -> dict:
        """
        Return a {movieId: cb_score} dict for every movie in the catalogue.
        Used by the fusion layer to blend with CF scores.
        """
        idx = self._find_movie_index(title)
        sim_row = self._sim_matrix[idx]
        return dict(zip(self.movies["movieId"].tolist(), sim_row.tolist()))

    def search(self, query: str) -> pd.DataFrame:
        """Return movies whose title contains the query string (case-insensitive)."""
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
        Resolve a title string to its DataFrame row index.
        Tries exact match first, then falls back to partial match.
        """
        title_lower = title.lower().strip()

        if title_lower in self._title_to_idx:
            return int(self._title_to_idx[title_lower])

        matches = self.movies[
            self.movies["title"].str.lower().str.contains(title_lower, na=False)
        ]
        if not matches.empty:
            matched_title = matches.iloc[0]["title"]
            print(f"[ContentBased] Partial match: '{title}' → '{matched_title}'")
            return int(matches.index[0])

        raise ValueError(
            f"Movie '{title}' not found. Use .search('{title}') to browse options."
        )
