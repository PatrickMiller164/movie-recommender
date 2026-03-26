import os
from dotenv import load_dotenv
load_dotenv()

import movie_recommender.config as c
from movie_recommender.backend.pipeline.extract import Extractor
from movie_recommender.backend.pipeline.transform import Transformer
from movie_recommender.backend.pipeline.recommend import recommend

API_KEY = os.environ['API_KEY']


def run_pipeline():
    print("Starting running pipeline")

    download_movie_universe = not c.MAIN_UNIVERSE_PARQUET.exists()

    if not c.EXTRACTED_PARQUET.exists():
        print("Running extractor")
        Extractor(API_KEY, download_movie_universe).run()

    if not c.TRANSFORMED_PARQUET.exists():
        print("Running transformer")
        Transformer().run()

    if not c.RECOMMENDATIONS_CSV.exists():
        print("Running recommender")
        recommend()

    print("Finished running pipeline.")

