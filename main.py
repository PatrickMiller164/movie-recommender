import os
import argparse
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

import config as c
from pipeline.extract import Extractor
from pipeline.transform import Transformer
from pipeline.recommend import recommend
from pipeline.recommend_similar import recommend_similar

API_KEY = os.environ['API_KEY']


class PipelineMode(Enum):
    FULL = "full"
    EXTRACT_ONLY = "extract_only"
    TRANSFORM_ONLY = 'transform_only'
    RECOMMEND = "recommend"
    RECOMMEND_SIMILAR = "recommend_similar"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run movie recommender in different modes"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in PipelineMode],
        default='recommend',
        help="Pipeline mode: full, extract_only, transform_only, recommend, recommend_similar"
    )
    parser.add_argument(
        "--download_movie_universe", 
        action="store_true",
        help="Download movie universe metadata from imdb (top 10000 films by number of votes)"
    )
    return parser.parse_args()


def main(mode: PipelineMode, download_movie_universe: bool):
    if mode in [PipelineMode.FULL, PipelineMode.EXTRACT_ONLY]:
        Extractor(API_KEY, download_movie_universe).run()
    
    if mode in [PipelineMode.FULL, PipelineMode.TRANSFORM_ONLY]:
        Transformer().run()

    if mode in [PipelineMode.FULL, PipelineMode.RECOMMEND]:
        recommend()

    if mode in [PipelineMode.RECOMMEND_SIMILAR]:
        recommend_similar()

    print("Finished")


if __name__=="__main__":
    args = parse_args()
    mode = PipelineMode(args.mode)
    download_movie_universe = args.download_movie_universe
    if download_movie_universe and mode not in [PipelineMode.FULL, PipelineMode.EXTRACT_ONLY]:
        print("Warning: -mode must be either 'full' or 'extract_only' for --download_movie_universe to take effect")

    if not c.TRANSFORMED_PARQUET.exists() or not c.RECOMMENDATIONS_CSV.exists():
        raise FileNotFoundError(
            f"Required files are missing:\n"
            f" - {c.TRANSFORMED_PARQUET}\n"
            f" - {c.RECOMMENDATIONS_CSV}\n"
            "Please run the main pipeline first (--mode full)"
        )

    print(f"Pipeline mode: {mode}")
    print(f"Download movie universe: {download_movie_universe}")

    main(mode, download_movie_universe)
