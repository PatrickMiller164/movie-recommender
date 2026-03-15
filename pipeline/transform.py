import polars as pl
from scipy.sparse import spmatrix
from polars.exceptions import ColumnNotFoundError
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import config as c
import numpy as np
from sklearn.cluster import KMeans

class Transformer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        df = pl.read_parquet(self.input_path)

        df = self._preprocess(df)
        df = self._set_nulls(df)
        df = self._rename_cols(df)
        df = self._transform_ratings(df)
        df = self._transform_misc(df)

        tfidf_matrix = self._generate_tfidf_document_matrix(df)
        df = self._run_clustering(df, tfidf_matrix)

        df.write_parquet(self.output_path)
        save_npz(c.PROJECT_ROOT/'data'/'tfidf_matrix.npz', tfidf_matrix)
        print("Finished running transformer")

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .filter(~pl.all_horizontal(pl.col(pl.Utf8).is_null()))
            .filter(pl.col('Type') == 'movie')
            .drop(['Title_0', 'totalSeasons', 'Response', 'Website', 'DVD', 'Production'])
        )
        return df
    
    def _set_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .with_columns([
                pl.col(c).replace("N/A", None)
                for c in df.columns if df[c].dtype == pl.Utf8
            ])
        )
        return df

    def _rename_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_dict = {
            'imdbrating': 'imdb_rating',
            'imdbvotes': 'imdb_votes',
            'imdbid': 'imdb_id',
            'boxoffice': 'box_office',
        }
        df = (
            df
            .rename({c: c.lower() for c in df.columns})
            .rename(rename_dict)
        )
        return df

    def _transform_ratings(self, df: pl.DataFrame) -> pl.DataFrame:
        rating_cols = ['rating_internet_movie_database', 'rating_rotten_tomatoes', 'rating_metacritic', 'rating_imdb','rating_metascore']
        df = (
            df
            .with_columns([
                (pl.col(c).str.split('/').list.get(0).cast(pl.Float32) / pl.col(c).str.split('/').list.get(1).cast(pl.Float32))
                .round(2) for c in ['rating_internet_movie_database', 'rating_metacritic']
            ]) 
            .with_columns([
                (pl.col('rating_rotten_tomatoes').str.split('%').list.get(0).cast(pl.Int16) / pl.lit(100)).round(2).alias('rating_rotten_tomatoes'),
                (pl.col('imdb_rating').cast(pl.Float32) / pl.lit(10)).round(2).alias('rating_imdb'),
                (pl.col('metascore').cast(pl.Int16) / pl.lit(100)).round(2).alias('rating_metascore')
            ])
            .with_columns(
                pl.mean_horizontal(rating_cols).round(3).alias('rating_mean')
            )
            .drop(['metascore', 'imdb_rating'])
        )
        return df        

    def _transform_misc(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (df
            .with_columns([
                expr for c in ['actors', 'language', 'country', 'genre', 'director', 'writer']
                for expr in [
                    pl.col(c).str.split(', ').list.eval(pl.element().str.replace_all(" ", "_")).alias(f"{c}_list"),
                    pl.col(c).str.split(', ').list.get(0).alias(f"primary_{c}")
                ]
            ])
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
            .drop(['runtime'])
        )

        cols = [c for c in df.columns if c != 'imdb_id']
        if 'imdb_id' in df.columns:
            df = df[['imdb_id'] + cols]
        
        return df

    def _generate_tfidf_document_matrix(self, df: pl.DataFrame) -> spmatrix:
        # Convert List(String) columns back to String columns
        document_cols=['actors_list', 'genre_list', 'director_list', 'writer_list', 'plot']
        for c in document_cols:
            if c not in df.columns:
                raise ColumnNotFoundError(f"'{c}' not in DataFrame, check spelling.")
            
            if isinstance(df[c].dtype, pl.List):
                df = df.with_columns(
                    pl.col(c).list.join(' ').alias(c)
                )

        # Create list of movie 'documents'
        plots = (
            df
            .select(pl.concat_str(document_cols, separator=' ').fill_null(""))
            .to_series()
            .to_list()
        )

        # Convert movie documents into numeric vectors
        vectoriser = TfidfVectorizer(
            stop_words='english', # Ignore common words like 'the', 'is', 'of'
            max_features=8000, # Limit vocabulary to top 5000 words by TF-IDF weight
            ngram_range=(1,2) # Setting the upper bound to 2 ensures phrases like 'serial killer' are captured as tokens
        )

        # Each plot is converted into a vector of 8000 TF-IDF features
        return vectoriser.fit_transform(plots)

    def _run_clustering(self, df: pl.DataFrame, tfidf_matrix: spmatrix):
        kmeans = KMeans(
            n_clusters=120,
            random_state=42,
            n_init=50
        )
        clusters = kmeans.fit_predict(tfidf_matrix)
        return df.with_columns(pl.Series(name='cluster', values=clusters))
    
                