import polars as pl
from sklearn.cluster import KMeans
from scipy.sparse import spmatrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from movie_recommender.config import Config


class Transformer:
    def __init__(self, config: Config):
        self.config = config
        self.vectoriser = TfidfVectorizer(
            stop_words='english',           # Ignore common words like 'the', 'is', 'of'
            max_features=8000,              # Limit vocabulary to top 5000 words by TF-IDF weight
            ngram_range=(1,2)               # Setting the upper bound to 2 ensures phrases like 'serial killer' are captured as tokens
        )
        self.model = KMeans(
            n_clusters=120, 
            random_state=42, 
            n_init=50
        )

    def run(self) -> None:
        df = pl.read_parquet(self.config.extracted_parquet)

        df = self._preprocess(df)
        df = self._set_nulls(df)
        df = self._rename_cols(df)
        df = self._transform_ratings(df)
        df = self._transform_misc(df)

        tfidf_matrix = self._generate_tfidf(df)
        self._persist_artefacts(tfidf_matrix)

        df = self._cluster(df, tfidf_matrix)

        self._write_outputs(df)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .filter(~pl.all_horizontal(pl.col(pl.Utf8).is_null()))
            .filter(pl.col('Type') == 'movie')
            .drop(['totalSeasons', 'Response', 'Website', 'DVD', 'Production'])
        )
    
    def _set_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col(c).replace("N/A", None).alias(c)
            for c in df.columns if df[c].dtype == pl.Utf8
        ])

    def _rename_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_dict = {
            'imdbrating': 'imdb_rating',
            'imdbvotes': 'imdb_votes',
            'imdbid': 'imdb_id',
            'boxoffice': 'box_office',
        }
        return df.rename({c: c.lower() for c in df.columns}).rename(rename_dict)

    def _safe_divide(self, num: pl.Expr, denom: pl.Expr) -> pl.Expr:
        return pl.when(denom != 0).then(num / denom).otherwise(None)

    def _transform_ratings(self, df: pl.DataFrame) -> pl.DataFrame:
        def parse_fraction(col: str) -> pl.Expr:
            split = pl.col(col).str.split("/")
            return self._safe_divide(
                split.list.get(0).cast(pl.Float32),
                split.list.get(1).cast(pl.Float32)
            )

        return (
            df
            .with_columns([
                parse_fraction("rating_internet_movie_database").round(2),
                parse_fraction("rating_metacritic").round(2)
            ])
            .with_columns([
                (pl.col('rating_rotten_tomatoes').str.replace("%","").cast(pl.Float32) / 100).round(2).alias('rating_rotten_tomatoes'),
                (pl.col('imdb_rating').cast(pl.Float32) / 10).round(2).alias('rating_imdb'),
                (pl.col('metascore').cast(pl.Int16) / 100).round(2).alias('rating_metascore')
            ])
            .with_columns([
                pl.mean_horizontal([
                    'rating_internet_movie_database', 
                    'rating_rotten_tomatoes', 
                    'rating_metacritic', 
                    'rating_imdb',
                    'rating_metascore'
                ])
                .round(3)
                .alias('rating_mean')
            ])
            .drop(['metascore', 'imdb_rating'], strict=False)
        )  

    def _transform_misc(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            expr for c in ['actors', 'language', 'country', 'genre', 'director', 'writer'] 
            for expr in [
                pl.col(c).str.split(', ').list.eval(pl.element().str.replace_all(" ", "_")).alias(f"{c}_list"),
                pl.col(c).str.split(', ').list.get(0).alias(f"primary_{c}")
            ]
        ])

        df = (
            df
            .with_columns([
                pl.col('runtime').str.extract(r"(\d+\.?\d*)").cast(pl.Int16).alias('runtime_mins'),
                pl.col('box_office').str.replace_all(r"[$,]", "").cast(pl.Int64),
                pl.col('released').str.to_date('%d %b %Y'),
                pl.col('year').cast(pl.Int16),
                pl.col('imdb_votes').str.replace_all(",","").cast(pl.Int32)
            ])
            .with_columns([
                ((pl.col('runtime_mins') // 5) * 5).alias('runtime_bucket'),
                ((pl.col('year') // 10) * 10).alias('decade')
            ])
            .drop(['runtime'], strict=False)
        )

        cols = [c for c in df.columns if c != 'imdb_id']
        if 'imdb_id' in df.columns:
            df = df[['imdb_id'] + cols]
        
        return df

    def _generate_tfidf(self, df: pl.DataFrame) -> spmatrix:
        # Convert List(String) columns back to String columns
        document_cols = ['actors_list', 'genre_list', 'director_list', 'writer_list', 'plot']

        df = df.with_columns([
            pl.when(pl.col(c).is_not_null())
            .then(pl.col(c).list.join(" ") if isinstance(df[c].dtype, pl.List) else pl.col(c))
            .otherwise(pl.lit(""))
            .alias(c)
            for c in document_cols
        ])

        # Create list of movie 'documents'
        plots = (
            df
            .select(pl.concat_str(document_cols, separator=' '))
            .to_series()
            .to_list()
        )

        # Convert movie documents into numeric vectors (8000 TF-IDF features each)
        return self.vectoriser.fit_transform(plots)

    def _cluster(self, df: pl.DataFrame, tfidf_matrix: spmatrix) -> pl.DataFrame:
        clusters = self.model.fit_predict(tfidf_matrix)
        return df.with_columns(pl.Series('cluster', clusters))
    
    def _persist_artefacts(self, matrix: spmatrix) -> None:
        save_npz(self.config.tfidf_maxtrix_npz, matrix)

    def _get_seen_movies_by_cluster(self, df: pl.DataFrame) -> pl.DataFrame:
        clusters = df['cluster'].unique().to_list()
        seen = df.filter(pl.col('watched'))
        dfs = [seen.filter(pl.col('cluster') == c).select('title').rename({'title': str(c)}) for c in clusters]
        dfs = [df for df in dfs if not df.is_empty()]
        return pl.concat(dfs, how='horizontal')

    def _write_outputs(self, df: pl.DataFrame) -> None:
        df.write_parquet(self.config.transformed_parquet)
        df.select([c for c in df.columns if df[c].dtype != pl.List]).write_csv(self.config.movie_universe_csv)
        self._get_seen_movies_by_cluster(df).write_csv(self.config.your_movies_by_cluster_csv)
    

if __name__=="__main__":
    import movie_recommender.config as c

    config = Config()
    Transformer(config).run() 