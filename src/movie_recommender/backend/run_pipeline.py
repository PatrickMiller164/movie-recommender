import os
from dotenv import load_dotenv
load_dotenv()

from movie_recommender.config import Config
from movie_recommender.backend.pipeline.extract import Extractor, OMDbClient
from movie_recommender.backend.pipeline.transform import Transformer
from movie_recommender.backend.pipeline.recommend import recommend

API_KEY = os.environ['API_KEY']


def run_pipeline():
    print("Starting running pipeline")

    config = Config()

    if not config.extracted_parquet.exists():
        print("Running extractor")
        client = OMDbClient(API_KEY)
        Extractor(config, client).run()

    if not config.transformed_parquet.exists():
        print("Running transformer")
        Transformer(config).run()

    if not config.recommendations_csv.exists():
        print("Running recommender")
        recommend()

    print("Finished running pipeline.")
