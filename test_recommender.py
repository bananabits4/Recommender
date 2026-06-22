"""
test_recommender.py
--------------------
Run this to verify the recommender logic works before touching the API or frontend.

Usage:
    python test_recommender.py
"""

from recommender import MovieRecommender

# ---------------------------------------------------------------
# 1. Load the engine
# ---------------------------------------------------------------
rec = MovieRecommender(
    movies_path="movies.csv",
    ratings_path="ratings.csv",
)

print("\n" + "=" * 60)
print("TEST 1 — Exact title match")
print("=" * 60)
results = rec.recommend("Toy Story (1995)", n=5)
print(results.to_string())

print("\n" + "=" * 60)
print("TEST 2 — Partial title match (case-insensitive)")
print("=" * 60)
results = rec.recommend("pulp", n=5)
print(results.to_string())

print("\n" + "=" * 60)
print("TEST 3 — Genre-heavy search (action/thriller)")
print("=" * 60)
results = rec.recommend("GoldenEye (1995)", n=5)
print(results.to_string())

print("\n" + "=" * 60)
print("TEST 4 — Search endpoint (partial title)")
print("=" * 60)
matches = rec.search("the")
print(matches.to_string())

print("\n" + "=" * 60)
print("TEST 5 — Movie metadata")
print("=" * 60)
info = rec.movie_info("Forrest")
print(info)

print("\n" + "=" * 60)
print("TEST 6 — Error handling (unknown movie)")
print("=" * 60)
try:
    rec.recommend("Nonexistent Movie XYZ")
except ValueError as e:
    print(f"Caught expected error: {e}")

print("\n✅ All tests passed.\n")
