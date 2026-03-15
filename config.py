import toml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

_config_file = PROJECT_ROOT / "config.toml"
_config = toml.load(_config_file)

FILMS_CSV = PROJECT_ROOT / _config['data']['films_csv']
RATINGS_TSV = PROJECT_ROOT / _config['data']['ratings_tsv']
MAIN_UNIVERSE_PARQUET = PROJECT_ROOT / _config['data']['main_universe_parquet']
EXTRACTED_PARQUET = PROJECT_ROOT / _config['data']['extracted_parquet']
TRANSFORMED_PARQUET = PROJECT_ROOT / _config['data']['transformed_parquet']

MOVIE_UNIVERSE_CSV = PROJECT_ROOT / _config['output']['movie_universe_csv']
YOUR_MOVIES_BY_CLUSTER = PROJECT_ROOT / _config['output']['your_movies_by_cluster_csv']
RECOMMENDATIONS_CSV = PROJECT_ROOT / _config['output']['recommendations_csv']
PL_RECOMMENDATIONS_CSV = PROJECT_ROOT / _config['output']['primary_language_recommendations_csv']
FL_RECOMMENDATIONS_CSV = PROJECT_ROOT / _config['output']['foreign_language_recommendations_csv']
