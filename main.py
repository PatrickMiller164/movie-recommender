import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.extract import Extractor
from pipeline.transform import Transformer

API_KEY = os.environ['API_KEY']

MAIN_UNIVERSE_EXTRACTED_PATH = 'data/main_universe_extracted.parquet'
FILMS_EXTRACTED_PATH = 'data/films_extracted.parquet'
FILMS_TRANSFORMED_PATH = 'data/films_transformed.parquet'

RETRIEVE_MAIN_UNIVERSE = False
RUN_EXTRACTOR = True
RUN_TRANSFORMER = True

def main():
    if RUN_EXTRACTOR:
        Extractor(API_KEY, RETRIEVE_MAIN_UNIVERSE, MAIN_UNIVERSE_EXTRACTED_PATH, FILMS_EXTRACTED_PATH).run()
    
    if RUN_TRANSFORMER:
        Transformer(FILMS_EXTRACTED_PATH, FILMS_TRANSFORMED_PATH).run()


if __name__=="__main__":
    main()