import toml
from pathlib import Path
from dataclasses import dataclass

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]

_config_file = REPO_ROOT / "config.toml"
_config = toml.load(_config_file)

FILMS_CSV = REPO_ROOT / _config['data']['films_csv']
RATINGS_TSV = REPO_ROOT / _config['data']['ratings_tsv']
MAIN_UNIVERSE_PARQUET = REPO_ROOT / _config['data']['main_universe_parquet']
EXTRACTED_PARQUET = REPO_ROOT / _config['data']['extracted_parquet']

TRANSFORMED_PARQUET = REPO_ROOT / _config['data']['transformed_parquet']
TFIDF_MATRIX_NPZ = REPO_ROOT / _config['data']['tfidf_matrix_npz']
YOUR_MOVIES_BY_CLUSTER_CSV = REPO_ROOT / _config['output']['your_movies_by_cluster_csv']
MOVIE_UNIVERSE_CSV = REPO_ROOT / _config['output']['movie_universe_csv']

RECOMMENDATIONS_CSV = REPO_ROOT / _config['output']['recommendations_csv']


@dataclass
class Config:
    users_films_csv: Path = FILMS_CSV
    imdb_ratings_tsv: Path = RATINGS_TSV
    main_universe_parquet: Path = MAIN_UNIVERSE_PARQUET
    extracted_parquet: Path = EXTRACTED_PARQUET
    transformed_parquet: Path = TRANSFORMED_PARQUET
    tfidf_maxtrix_npz: Path = TFIDF_MATRIX_NPZ
    your_movies_by_cluster_csv: Path = YOUR_MOVIES_BY_CLUSTER_CSV
    movie_universe_csv: Path = MOVIE_UNIVERSE_CSV
    recommendations_csv: Path = RECOMMENDATIONS_CSV
