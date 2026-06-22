"""
Flask API — Movie Recommender
------------------------------
Exposes the recommender engine as a REST API.
The HTML frontend (frontend/) will call these endpoints.

Endpoints:
    GET  /recommend?title=<title>&n=<n>   → top-N recommendations
    GET  /search?q=<query>                → partial title search
    GET  /movie?title=<title>             → single movie metadata
    GET  /health                          → sanity check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from recommender import MovieRecommender

app = Flask(__name__)
CORS(app)  # Allow the HTML frontend (different port / file://) to call us

# Load once at startup — not on every request
recommender = MovieRecommender(
    movies_path="movies.csv",
    ratings_path="ratings.csv",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "movies_loaded": len(recommender.movies)})


@app.get("/recommend")
def recommend():
    """
    Query params:
        title (str, required) — movie title to base recommendations on
        n     (int, optional) — number of results (default 5, max 20)
    """
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing 'title' query parameter."}), 400

    n = min(int(request.args.get("n", 5)), 20)  # cap at 20

    try:
        results = recommender.recommend(title, n=n)
        return jsonify({
            "query": title,
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
        title (str, required) — exact or partial title
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
