import polars as pl


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

        df.write_parquet(self.output_path)
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
