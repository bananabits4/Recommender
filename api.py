"""
Flask API — Hybrid Movie Recommender
--------------------------------------
Exposes the hybrid recommender engine as a REST API.
The HTML frontend (frontend/) calls these endpoints.

Endpoints:
    GET  /recommend?title=<title>&user_id=<id>&n=<n>&alpha=<alpha>
                                              → top-N hybrid recommendations
    GET  /search?q=<query>                    → partial title search
    GET  /movie?title=<title>                 → single movie metadata
    GET  /health                              → sanity check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.fusion import HybridRecommender

app = Flask(__name__)
CORS(app)  # Allow the HTML frontend (different port / file://) to call us

# Load and fit both sub-models once at startup — not on every request
recommender = HybridRecommender(
    movies_path="data/movies.csv",
    ratings_path="data/ratings.csv",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "movies_loaded": len(recommender.movies),
    })


@app.get("/recommend")
def recommend():
    """
    Query params:
        title   (str, optional) — seed movie for content-based signal
        user_id (int, optional) — user for collaborative signal
        n       (int, optional) — number of results (default 5, max 20)
        alpha   (float, optional) — CF weight in [0,1]; auto-chosen if omitted

    At least one of title or user_id must be provided.
    """
    title = request.args.get("title", "").strip() or None
    user_id = request.args.get("user_id", None)
    n = min(int(request.args.get("n", 5)), 20)


    try:
        results = recommender.recommend(title=title, user_id=user_id, n=n, alpha=0.5)
        return jsonify({
            "query": {"title": title, "user_id": user_id, "alpha": 0.5},
            "n": n,
            "recommendations": results.to_dict(orient="records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.get("/search")
def search():
    """
    Query params:
        q (str, required) — partial title to search for
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter."}), 400

    results = recommender.search(query)
    return jsonify({
        "query": query,
        "results": results.to_dict(orient="records"),
    })


@app.get("/movie")
def movie_info():
    """
    Query params:
        title (str, required) — exact or partial movie title
    """
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing 'title' query parameter."}), 400

    try:
        info = recommender.movie_info(title)
        return jsonify(info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=True gives auto-reload during development — turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
