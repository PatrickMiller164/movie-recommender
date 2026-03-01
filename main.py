import os
from dotenv import load_dotenv
load_dotenv()

from config import FILMS_CSV, RATINGS_TSV, MAIN_UNIVERSE_PARQUET, EXTRACTED_PARQUET, TRANSFORMED_PARQUET, OMDB_BASE_URL
from pipeline.extract import Extractor
from pipeline.transform import Transformer

API_KEY = os.environ['API_KEY']

RETRIEVE_MAIN_UNIVERSE = False
RUN_EXTRACTOR = True
RUN_TRANSFORMER = True

def main():
    if RUN_EXTRACTOR:
        Extractor(API_KEY, RETRIEVE_MAIN_UNIVERSE, FILMS_CSV, RATINGS_TSV, MAIN_UNIVERSE_PARQUET, EXTRACTED_PARQUET, OMDB_BASE_URL).run()
    
    if RUN_TRANSFORMER:
        Transformer(EXTRACTED_PARQUET, TRANSFORMED_PARQUET).run()


if __name__=="__main__":
    main()