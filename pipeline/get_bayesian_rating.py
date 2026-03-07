import polars as pl

def get_bayesian_rating(universe: pl.DataFrame) -> pl.DataFrame:
    C = universe['rating_mean'].mean()
    m = universe['imdb_votes'].quantile(0.75)

    universe = (
        universe
        .with_columns(
            ((pl.col('imdb_votes') / (pl.col('imdb_votes') + m)) * pl.col('rating_mean') +
             (m / (pl.col('imdb_votes') + m)) * C
            ).alias('bayesian_rating')
        )
    )
    return universe.select(['imdb_id', 'bayesian_rating'])