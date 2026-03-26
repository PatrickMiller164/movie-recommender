import os
from dotenv import load_dotenv
load_dotenv()

import movie_recommender.config as c
from movie_recommender.backend.pipeline.extract import Extractor, OMDbClient
from movie_recommender.backend.pipeline.transform import Transformer
from movie_recommender.backend.pipeline.recommend import recommend

API_KEY = os.environ['API_KEY']


def run_pipeline():
    print("Starting running pipeline")

    if not c.EXTRACTED_PARQUET.exists():
        print("Running extractor")
        client = OMDbClient(API_KEY)
        Extractor(client).run()

    if not c.TRANSFORMED_PARQUET.exists():
        print("Running transformer")
        Transformer().run()

    if not c.RECOMMENDATIONS_CSV.exists():
        print("Running recommender")
        recommend()

    print("Finished running pipeline.")

