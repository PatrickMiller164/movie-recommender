from fastapi import FastAPI
import polars as pl

from pipeline.recommend_similar import fuzzy_match, found_exact_match, get_similar, get_top_three
import config as c

app = FastAPI(title="Movie Recommender API")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": {item_id}}


@app.post("/recommend_similar/{movie}")
async def recommend_similar(movie: str):
    movie_universe = pl.read_parquet(c.TRANSFORMED_PARQUET)
    recommendations = pl.read_csv(c.RECOMMENDATIONS_CSV)
    all_movies = movie_universe['title'].to_list()
    results = fuzzy_match(movie, all_movies)
    if not found_exact_match(results):
        return {"message": f"Did you mean one of the following: {[r[0] for r in results]}"}

    similar = get_similar(movie_universe, recommendations, movie)
    top_three = get_top_three(similar)

    message = f"Top 3 similar movies: {', '.join(top_three)}"
    return {"message": message}
