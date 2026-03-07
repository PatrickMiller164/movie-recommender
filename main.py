import os
import polars as pl
import argparse
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

from config import FILMS_CSV, RATINGS_TSV, MAIN_UNIVERSE_PARQUET, EXTRACTED_PARQUET, TRANSFORMED_PARQUET, PRIMARY_LANGUAGE, PL_RECOMMENDATIONS_CSV, FL_RECOMMENDATIONS_CSV
from pipeline.extract import Extractor
from pipeline.transform import Transformer
from pipeline.method_simple_composite import run_simple_composite
from pipeline.method_vector_similarity import run_vector_similarity
from pipeline.method_tfidf_plot_similarity import run_tfidf_plot_similarity
from pipeline.get_bayesian_rating import get_bayesian_rating

API_KEY = os.environ['API_KEY']

RETRIEVE_MAIN_UNIVERSE = False


class PipelineMode(Enum):
    FULL = "full"
    SCORE_ONLY = "score_only"
    EXTRACT_ONLY = "extract_only"
    TRANSFORM_ONLY = 'transform_only'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run movie recommender in different modes"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in PipelineMode],
        default='score_only',
        help="Pipeline mode: full, score_only, extract_only, transform_only"
    )
    parser.add_argument(
        "--download_movie_universe", 
        action="store_true",
        help="Download movie universe metadata from imdb (top 10000 films by number of votes)"
    )
    return parser.parse_args()


def main(mode: PipelineMode, download_movie_universe: bool):
    if mode in [PipelineMode.FULL, PipelineMode.EXTRACT_ONLY]:
        Extractor(
            API_KEY, 
            download_movie_universe, 
            FILMS_CSV, 
            RATINGS_TSV, 
            MAIN_UNIVERSE_PARQUET, 
            EXTRACTED_PARQUET
        ).run()
    
    if mode in [PipelineMode.FULL, PipelineMode.TRANSFORM_ONLY]:
        Transformer(EXTRACTED_PARQUET, TRANSFORMED_PARQUET).run()

    if mode in [PipelineMode.FULL, PipelineMode.SCORE_ONLY]:
        universe = pl.read_parquet(TRANSFORMED_PARQUET)
        unseen = universe.filter(~pl.col('watched'))
        favourites = universe.filter(pl.col('favourites'))

        m1 = run_simple_composite(unseen, favourites)
        m2 = run_vector_similarity(universe, unseen, favourites)
        m3 = run_tfidf_plot_similarity(
            universe, 
            unseen, 
            favourites, 
            document_cols=['actors_list', 'genre_list', 'director_list', 'writer_list', 'plot']
        )
        stabilised_score = get_bayesian_rating(universe)

        for df in [m1, m2, m3, stabilised_score]:
            unseen = unseen.join(df, on='imdb_id', how='left')

        # Create recommendations
        # - Filter for films with above average mean rating
        # - Sort by a similarity score, best recommendations first
        # - Break up into primary language and others, then export
        output_cols = [
            'imdb_id', 'title', 'year', 'genre', 'rating_mean', 'imdb_votes', 'simple_composite_score', 
            'vector_similarity', 'tfidf_document_similarity', 'bayesian_rating',
            'runtime_mins',  'primary_language', 'primary_country',  'director', 'writer', 'actors', 'plot'
        ]
        unseen = (
            unseen
            .filter(pl.col('rating_mean') > unseen['bayesian_rating'].mean())
            .sort('tfidf_document_similarity', descending=True, nulls_last=True)
            .select(output_cols)
        )

        l = PRIMARY_LANGUAGE if PRIMARY_LANGUAGE in unseen['primary_language'].to_list() else 'English'
        unseen.filter(pl.col('primary_language') == l).write_csv(PL_RECOMMENDATIONS_CSV)
        unseen.filter(pl.col('primary_language') != l).write_csv(FL_RECOMMENDATIONS_CSV)
            
        print("Finished running scorer")
        
    print("Finished")


if __name__=="__main__":
    args = parse_args()
    mode = PipelineMode(args.mode)
    download_movie_universe = args.download_movie_universe
    if download_movie_universe and mode not in [PipelineMode.FULL, PipelineMode.EXTRACT_ONLY]:
        print("Warning: -mode must be either 'full' or 'extract_only' for --download_movie_universe to take effect")

    print(f"Pipeline mode: {mode}")
    print(f"Download movie universe: {download_movie_universe}")

    main(mode, download_movie_universe)

