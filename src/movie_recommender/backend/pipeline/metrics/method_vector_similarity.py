import polars as pl
import numpy as np

multi_valued_features = ['director_list', 'writer_list', 'actors_list', 'genre_list', 'country_list', 'language_list']


def create_one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    for col in multi_valued_features:
        ohe = (
            df
            .select(['imdb_id', col])
            .explode(col)
            .with_columns(pl.lit(1).alias('value'))
            .pivot(values='value', index='imdb_id', on=col, aggregate_function='max')
            .fill_null(0)
        )

        ohe = ohe.rename({c: f'{col.removesuffix("_list")}_{c}' for c in ohe.columns if c != 'imdb_id'})
        df = df.join(ohe, on='imdb_id', how='left')
    return df


def compute_similarity(df: pl.DataFrame, favourites: pl.DataFrame) -> pl.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith(("director_", "writer_", "actors_", "genre_")) and c not in multi_valued_features]

    X = df.select(feature_cols).to_numpy()
    taste_vector = favourites.select(feature_cols).mean().row(0)
    taste_vec = np.array(taste_vector)

    sin_scores = (X @ taste_vec) / (np.linalg.norm(X, axis=1) * np.linalg.norm(taste_vec) + 1e-8)

    df = (
        df
        .with_columns(pl.Series('vector_similarity', sin_scores).round(3))
        .fill_nan(0)
        .sort('vector_similarity', descending=True, nulls_last=True)
    )
    return df


def run_vector_similarity(universe: pl.DataFrame, unseen: pl.DataFrame, favourites: pl.DataFrame) -> pl.DataFrame:
    universe = create_one_hot_encoding(universe)

    unseen = universe.filter(pl.col('imdb_id').is_in(unseen['imdb_id']))
    favourites = universe.filter(pl.col('imdb_id').is_in(favourites['imdb_id']))

    unseen = compute_similarity(unseen, favourites)
    return unseen.select(['imdb_id', 'vector_similarity'])