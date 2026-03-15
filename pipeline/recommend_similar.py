import config as c
import polars as pl

from rapidfuzz import process, fuzz


def fuzzy_match(input_string, choices):
    return process.extract(
        query=input_string,
        choices=choices,
        scorer=fuzz.ratio,
        limit=5
    )


def found_exact_match(results):
    return results[0][1] == 100.0


def find_exact_match(all_movies):
    while True:
        user_input = input("Enter a movie: ")
        results = fuzzy_match(user_input, all_movies)
        if found_exact_match(results):
            return user_input
        else:
            print(f"Did you mean one of the following: {[r[0] for r in results]}")


def get_top_three(df: pl.DataFrame) -> list:
    return (
        df
        .head(3)
        .select(
            pl.concat_str([
                pl.col('title'), 
                pl.lit(' ('), 
                pl.col('year').cast(pl.Utf8), 
                pl.lit(')')
            ])
        )
        .to_series()
        .to_list()
    )

def recommend_similar():
    movie_universe = pl.read_parquet(c.TRANSFORMED_PARQUET)
    recommendations = pl.read_csv(c.RECOMMENDATIONS_CSV)
    all_movies = movie_universe['title'].to_list()

    movie = find_exact_match(all_movies)
    movie_cluster = movie_universe.filter(pl.col('title') == movie)['cluster'].to_list()[0]

    res = recommendations.filter(pl.col('cluster') == movie_cluster).head(30)

    relative_path = f'output/movies similar to {movie}.csv'
    res.write_csv(c.PROJECT_ROOT/relative_path)

    top_three = get_top_three(res)

    print("\n🎬 Top 3 similar movies:")
    for i, title in enumerate(top_three, 1):
        print(f"{i}. {title}")

    print(f"\nFull list of recommendations saved at: {relative_path}")