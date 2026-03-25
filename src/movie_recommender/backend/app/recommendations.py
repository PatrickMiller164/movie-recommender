from typing import Optional
from dataclasses import dataclass
from enum import Enum

import polars as pl
from fastapi import APIRouter

import movie_recommender.config as c
from movie_recommender.backend.run_pipeline import run_pipeline

router = APIRouter()


@dataclass(frozen=True)
class SortInfo():
    display_name: str
    field: str
    descending: bool


class SortBy(Enum):
    RECOMMENDATION_SCORE_DESCENDING = SortInfo("Recommendation Score (descending)", "score", True)
    ONLINE_RATING_DESCENDING = SortInfo("Online Rating (descending)", "rating_bayesian", True)
    NUMBER_OF_VOTES_DESCENDING = SortInfo("Number of Votes (descending)", "imdb_votes", True)
    RUNTIME_DESCENDING = SortInfo("Runtime (descending)", "runtime_mins", True)
    RUNTIME_ASCENDING = SortInfo("Runtime (ascending)", "runtime_mins", False)


SORT_MAP = {s.value.display_name: s for s in SortBy}


def load_recommendations() -> pl.DataFrame:
    if not c.RECOMMENDATIONS_CSV.exists():
        run_pipeline()

    columns = [
         'title', 'year', 'genre', 'rating_bayesian', 'imdb_votes', 'score',
         'runtime_mins', 'primary_language', 'primary_country', 'plot', 'poster',
         'director', 'writer', 'actors'
    ]
    return pl.read_csv(c.RECOMMENDATIONS_CSV, columns=columns)

recommendations_df = load_recommendations()


def filter_sort_df(df, member=SortBy.RECOMMENDATION_SCORE_DESCENDING, genre=None, language=None, country=None) -> pl.DataFrame:
    if genre:
        df = df.filter(pl.col('genre').str.split(", ").list.contains(genre))
    if language:
        df = df.filter(pl.col('primary_language') == language)
    if country:
        df = df.filter(pl.col('primary_country') == country)   
    df = df.sort(pl.col(member.value.field), descending=member.value.descending)
    return df


def get_field_options(df: pl.DataFrame, field: str, explode: bool = False) -> list[str]:
    series = df[field].str.split(", ").explode() if explode else df[field]
    return series.value_counts().sort('count', descending=True)[field].to_list()


@router.get("/recommendations")
async def get_recommendations(
    genre: Optional[str] = None,
    language: Optional[str] = None,
    country: Optional[str] = None,
    sort_by: str = SortBy.RECOMMENDATION_SCORE_DESCENDING.value.display_name,
    limit: int = 20
):
    member = SORT_MAP.get(sort_by, SortBy.RECOMMENDATION_SCORE_DESCENDING)

    df = filter_sort_df(recommendations_df, member, genre, language, country)

    all_genres = get_field_options(df, 'genre', True)
    all_countries = get_field_options(df, 'primary_country')
    all_languages = get_field_options(df, 'primary_language')
    all_sort_options = [s.value.display_name for s in SortBy]

    df_dicts = df.slice(0, limit).to_dicts()
    more_results_available = len(df) > len(df_dicts)

    return {
        "status": "success",
        "results": df_dicts,
        "all_genres": all_genres,
        "all_languages": all_languages,
        "all_countries": all_countries,
        "sort_options": all_sort_options,
        "default_sort_option": SortBy.RECOMMENDATION_SCORE_DESCENDING.value.display_name,
        "more_results_available": more_results_available
    }