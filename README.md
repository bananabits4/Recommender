# Movie Recommender System

A content-based movie recommender using TF-IDF + cosine similarity, served via a Flask REST API with a plain HTML/JS frontend.

Built with Python, scikit-learn, Flask, HTML/CSS/JS — no React, no build step.

---

## How it works

```
movies.csv (title + genres)
        ↓
  TF-IDF Vectorizer          ← genres + title words become feature vectors
        ↓
  Cosine Similarity Matrix   ← n_movies × n_movies similarity scores
        ↓
  recommend(title, n)        ← top-N most similar movies
        ↓
  Flask API (/recommend)     ← JSON response
        ↓
  HTML Frontend              ← user types a title, sees recommendations
```

The key insight: movies with overlapping genres and title keywords get high cosine similarity scores. "Toy Story (1995)" → Adventure|Animation|Children|Comedy|Fantasy will rank "Jumanji" and "Balto" above "Casino" because their genre vectors are closer.

---

## Project structure

```
movie-recommender/
├── data/
│   ├── movies.csv          ← movieId, title, genres
│   └── ratings.csv         ← userId, movieId, rating, timestamp  (for future CF step)
│
├── recommender.py          ← MovieRecommender class (pure ML, no web)
├── api.py                  ← Flask app wrapping the recommender
├── test_recommender.py     ← run this first to verify everything works
│
├── frontend/               ← (coming next)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas scikit-learn flask flask-cors
```

### 2. Run the tests (no server needed)

```bash
python test_recommender.py
```

### 3. Start the API

```bash
python api.py
# → Running on http://localhost:5000
```

### 4. Try the API

```bash
# Get 5 recommendations for Toy Story
curl "http://localhost:5000/recommend?title=Toy+Story&n=5"

# Search movies by partial title
curl "http://localhost:5000/search?q=dark"

# Get movie metadata
curl "http://localhost:5000/movie?title=Pulp+Fiction"

# Health check
curl "http://localhost:5000/health"
```

---

## API reference

| Endpoint | Params | Returns |
|---|---|---|
| `GET /recommend` | `title` (required), `n` (default 5) | Top-N similar movies with similarity scores |
| `GET /search` | `q` (partial title) | All matching movies |
| `GET /movie` | `title` | Single movie metadata |
| `GET /health` | — | Status + movie count |

### Example response — `/recommend?title=Heat&n=3`

```json
{
  "query": "Heat",
  "n": 3,
  "recommendations": [
    { "title": "Assassins (1995)", "genres": "Action|Crime|Thriller", "similarity_score": 0.4112 },
    { "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller", "similarity_score": 0.3258 },
    { "title": "Copycat (1995)",   "genres": "Crime|Drama|Horror|Mystery|Thriller", "similarity_score": 0.2533 }
  ]
}
```

---

## Resume talking points

- **TF-IDF + cosine similarity** on structured genre/title text — end-to-end content-based filtering
- **Scikit-learn pipeline** with a pre-computed similarity matrix for fast inference
- **REST API** with proper error handling, CORS, and query validation
- **Partial title matching** — users don't need exact movie titles
- Dataset: MovieLens format (50 movies, 2500+ ratings)

---

## What's next (to turn this into a hybrid system)

1. **Add collaborative filtering** — SVD on the user-item rating matrix using `surprise` library. Produces a CF score per (user, movie) pair.
2. **Hybrid fusion** — `final_score = α × CF_score + (1-α) × CB_score`. Tune α per user based on how many ratings they have (cold-start users lean content-based).
3. **C++ speedup** — Replace the NumPy dot product in cosine similarity with a C++ extension via `pybind11`. Benchmark and report the speedup in your README.
4. **Frontend** — Plain HTML form that calls `/recommend` and renders cards.
