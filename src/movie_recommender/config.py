import toml
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

REPO_ROOT = MODULE_ROOT.parents[1]

_config_file = REPO_ROOT / "config.toml"
_config = toml.load(_config_file)

FILMS_CSV = REPO_ROOT / _config['data']['films_csv']
RATINGS_TSV = REPO_ROOT / _config['data']['ratings_tsv']
MAIN_UNIVERSE_PARQUET = REPO_ROOT / _config['data']['main_universe_parquet']
EXTRACTED_PARQUET = REPO_ROOT / _config['data']['extracted_parquet']
TRANSFORMED_PARQUET = REPO_ROOT / _config['data']['transformed_parquet']

MOVIE_UNIVERSE_CSV = REPO_ROOT / _config['output']['movie_universe_csv']
YOUR_MOVIES_BY_CLUSTER = REPO_ROOT / _config['output']['your_movies_by_cluster_csv']
RECOMMENDATIONS_CSV = REPO_ROOT / _config['output']['recommendations_csv']
PL_RECOMMENDATIONS_CSV = REPO_ROOT / _config['output']['primary_language_recommendations_csv']
FL_RECOMMENDATIONS_CSV = REPO_ROOT / _config['output']['foreign_language_recommendations_csv']
