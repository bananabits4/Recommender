# Movie Recommender System

A content-based movie recommender using TF-IDF + cosine similarity and SVD based filtering, served via a Flask REST API with a plain HTML/JS frontend.

Built with Python, scikit-learn, Flask, HTML/CSS/JS — no React, no build step.

---



## Project structure

```
movie-recommender/
├── data/
│   ├── movies.csv          ← movieId, title, genres
│   └── ratings.csv         ← userId, movieId, rating, timestamp  (for future CF step)
├── src/
|   ├──collaborative.py          ← Collaborative filtering class
|   ├──contect_based.py          ← Content based filtering class
|   ├──fusion.py                 ← Combines both recommendations
├── api.py                   ← Flask app wrapping the recommender
├── index.html               ← HTML frontend for output display
│
└── README.md
```

---

## Quickstart

### Install dependencies

```bash
pip install pandas scikit-learn flask flask-cors
```

### Start the API

```bash
python api.py
# → Running on http://localhost:5000
```

### Run the HTML

Use the frontend and enjoy.


---
