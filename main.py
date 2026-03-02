import os
import polars as pl
from dotenv import load_dotenv
load_dotenv()

from config import FILMS_CSV, RATINGS_TSV, MAIN_UNIVERSE_PARQUET, EXTRACTED_PARQUET, TRANSFORMED_PARQUET, OMDB_BASE_URL, RECOMMENDATIONS_CSV
from pipeline.extract import Extractor
from pipeline.transform import Transformer
from pipeline.method_simple_composite import run_simple_composite
from pipeline.method_vector_similarity import run_vector_similarity

API_KEY = os.environ['API_KEY']

RETRIEVE_MAIN_UNIVERSE = False
RUN_EXTRACTOR = False
RUN_TRANSFORMER = False

def main():
    if RUN_EXTRACTOR:
        Extractor(API_KEY, RETRIEVE_MAIN_UNIVERSE, FILMS_CSV, RATINGS_TSV, MAIN_UNIVERSE_PARQUET, EXTRACTED_PARQUET, OMDB_BASE_URL).run()
    
    if RUN_TRANSFORMER:
        Transformer(EXTRACTED_PARQUET, TRANSFORMED_PARQUET).run()

    universe = pl.read_parquet(TRANSFORMED_PARQUET)
    unseen = universe.filter(~pl.col('watched'))
    favourites = universe.filter(pl.col('rating_me') >= 2)

    m1 = run_simple_composite(unseen, favourites)
    m2 = run_vector_similarity(universe, unseen, favourites)

    unseen = (
        unseen
        .join(m1, on='imdb_id', how='left')
        .join(m2, on='imdb_id', how='left')
    )

    output_cols = [
        'imdb_id', 'title', 'year', 'genre', 'director', 'writer', 'actors', 
        'plot', 'primary_language', 'primary_country', 'poster', 'imdb_votes', 
        'rating_mean', 'runtime_mins', 'simple_composite_score', 'vector_similarity'
    ]
    unseen.select(output_cols).write_csv(RECOMMENDATIONS_CSV)
    
    print("Finished")

if __name__=="__main__":
    main()

