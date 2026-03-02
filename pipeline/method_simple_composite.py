import polars as pl


def find_top(df: pl.DataFrame, col: str, explode: bool = False):
    if explode:
        df = (df
            .with_columns(
                pl.col(col)
                .str.split(',')
            )
            .explode(col)
        )

    score_col = f'{col}_score'
    df = (
        df
        .group_by(col)
        .agg([
            pl.mean('rating_me').alias('avg_rating'),
            pl.len().alias('count')
        ])
        .with_columns(
            (pl.col('avg_rating') * pl.col('count')).alias(score_col)
        )
        .sort(score_col, descending=True)
        .select([col, score_col])
    )
    return df


def apply_score(df: pl.DataFrame, col: str, top_df: pl.DataFrame):
    return (
    df
    .with_columns(pl.col(col).str.split(','))
    .select(['imdb_id', col])
    .explode(col)
    .join(top_df, on=col, how='left')
    .group_by('imdb_id')
    .agg([
        pl.max(f'{col}_score')
    ])
    .with_columns(f'{col}_score').fill_null(0)
)


def run_simple_composite(unseen: pl.DataFrame, favourites: pl.DataFrame) -> pl.DataFrame:
    top_directors = find_top(favourites, 'director', True)
    top_genres = find_top(favourites, 'genre').with_columns(pl.col('genre_score').log())
    top_actors = find_top(favourites, 'actors', True)
    top_writers = find_top(favourites, 'writer', True)

    unseen = (
        unseen
        .join(apply_score(unseen, 'director', top_directors), on='imdb_id', how='left')
        .join(apply_score(unseen, 'writer', top_writers), on='imdb_id', how='left')
        .join(apply_score(unseen, 'actors', top_actors), on='imdb_id', how='left')
        .join(top_genres, on='genre', how='left')
        .with_columns(
            pl.sum_horizontal(['director_score', 'writer_score', 'actors_score']).alias('simple_composite_score')
        )
    )
    return unseen.select(['imdb_id', 'simple_composite_score'])




