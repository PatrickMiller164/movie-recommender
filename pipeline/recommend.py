import polars as pl
import config as c
from scipy.sparse import load_npz

from pipeline.method_simple_composite import run_simple_composite
from pipeline.method_vector_similarity import run_vector_similarity
from pipeline.method_tfidf_plot_similarity import run_tfidf_plot_similarity
from pipeline.get_bayesian_rating import get_bayesian_rating


def create_final_score(df: pl.DataFrame, score_cols: list[str]) -> pl.DataFrame:
    return (
        df
        .with_columns([
            (
                (pl.col(c) - pl.col(c).min()) / 
                (pl.col(c).max() - pl.col(c).min())
            )
            .alias(f"{c}_normalised")
            for c in score_cols
        ])
        .with_columns(
            pl.mean_horizontal([pl.col(f"{c}_normalised") for c in score_cols]).round(3).alias('score')
        )
    )


def recommend() -> None:
    universe = pl.read_parquet(c.TRANSFORMED_PARQUET)
    unseen = universe.filter(~pl.col('watched'))
    favourites = universe.filter(pl.col('favourites'))
    tfidf_matrix = load_npz(c.PROJECT_ROOT/'data'/'tfidf_matrix.npz')

    # Calculate metrics
    m1 = run_simple_composite(unseen, favourites)
    print("Run simple composite")

    m2 = run_vector_similarity(universe, unseen, favourites)
    print("Run vector similarity")

    m3 = run_tfidf_plot_similarity(universe, unseen, favourites, tfidf_matrix)
    print("Run tfidf plot similarity ")

    m4 = get_bayesian_rating(universe)
    bayesian_rating_mean = m4['rating_bayesian'].mean()
    print("Run bayesian rating")

    # Join metrics to database
    metrics = [m1, m2, m3, m4]
    for m in metrics:
        unseen = unseen.join(m, on='imdb_id', how='left')

    # Create final score
    unseen = create_final_score(unseen, score_cols=['vector_similarity', 'tfidf_document_similarity'])

    # Create recommendations
    # - Filter for films with above average mean rating
    # - Sort by a similarity score, best recommendations first
    output_cols = [
        'imdb_id', 'title', 'year', 'genre', 'cluster', 'score', 'rating_mean', 'rating_bayesian', 'imdb_votes', 
        'simple_composite_score', 'vector_similarity', 'tfidf_document_similarity',
        'runtime_mins', 'primary_language', 'primary_country',  'director', 'writer', 'actors', 'plot'
    ]
    unseen = (
        unseen
        .filter(pl.col('rating_bayesian') > bayesian_rating_mean)
        .sort('score', descending=True, nulls_last=True)
        .select(output_cols)
    )

    # - Break up into primary language and others, then export
    unseen.write_csv(c.RECOMMENDATIONS_CSV)
    unseen.filter(pl.col('primary_language') == 'English').write_csv(c.PL_RECOMMENDATIONS_CSV)
    unseen.filter(pl.col('primary_language') != 'English').write_csv(c.FL_RECOMMENDATIONS_CSV)

    print("Finished running scorer")