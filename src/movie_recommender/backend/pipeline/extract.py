import requests
from concurrent.futures import ThreadPoolExecutor

import polars as pl

import movie_recommender.config as c


class OMDbClient:
    def __init__(self, api_key: str, base_url: str = "https://www.omdbapi.com"):
        self.api_key = api_key
        self.base_url = base_url

    def get_by_title(self, title: str) -> dict | None:
        return self._get_data(title=title)

    def get_by_id(self, imdb_id: str) -> dict | None:
        return self._get_data(imdb_id=imdb_id)

    def _get_data(self, *, title: str | None = None, imdb_id: str | None = None) -> dict | None:
        if (title is None) == (imdb_id is None):
            raise ValueError("Provide exactly one of title or imdb_id")

        data = self._make_request(title=title, imdb_id=imdb_id)
        if not self._found_title(data):
            print(f"No result for '{title}'")
            return None
        data = self._fix_ratings(data)
        return data

    def _make_request(self, title: str | None = None, imdb_id: str | None = None) -> dict:
        params = {'apikey': self.api_key}
        if title:
            params['t'] = title
        if imdb_id:
            params['i'] = imdb_id
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def _found_title(self, data: dict) -> bool:
        return data['Response'] == 'True'

    def _fix_ratings(self, data: dict) -> dict:
        for d in data.get('Ratings', []):
            source = d['Source']
            rating = d['Value']
            data[f'rating_{source.replace(" ", "_")}'] = rating
        data.pop('Ratings', None)
        return data


class Extractor:
    def __init__(self, client: OMDbClient, request_main_universe: bool = False):
        self.client = client
        self.request_main_universe = request_main_universe

        self.watched = set()
        self.favourites = set()

    def run(self):
        df = pl.read_csv(c.FILMS_CSV)
        self.watched = set(df['All'].drop_nulls())
        self.favourites = set(df['Favourites'].drop_nulls())

        main_universe_df = self._retrieve_main_universe()
        main_universe_titles = set(main_universe_df['Title'])

        remaining_titles = self.watched - main_universe_titles
        remaining_df = self._retrieve_remaining_titles(remaining_titles)

        films = pl.concat([main_universe_df, remaining_df.select(main_universe_df.columns)])
        films = self._enrich(films)
        films.write_parquet(c.EXTRACTED_PARQUET)

    def _retrieve_main_universe(self) -> pl.DataFrame:
        if not self.request_main_universe and c.MAIN_UNIVERSE_PARQUET.exists():
            return pl.read_parquet(c.MAIN_UNIVERSE_PARQUET)

        top_movie_ids = self._get_top_movie_ids(10000)
        with ThreadPoolExecutor(max_workers=10) as executor:
            records = [
                r for r in executor.map(self.client.get_by_id, top_movie_ids)
                if r is not None
            ]

        df = pl.from_dicts(records)
        df.write_parquet(c.MAIN_UNIVERSE_PARQUET)

        return df
    
    def _retrieve_remaining_titles(self, remaining: set[str]) -> pl.DataFrame:
        print(f"Fetching metadata for {len(remaining)} other movies")

        with ThreadPoolExecutor(max_workers=10) as executor:
            records = [
                r for r in executor.map(self.client.get_by_title, remaining)
                if r is not None
            ]

        return pl.from_dicts(records)
    
    def _get_top_movie_ids(self, num: int = 10000) -> set[str]:
        df = self._read_tsv(c.RATINGS_TSV)
        top = df.sort('numVotes', descending=True).head(num)
        return set(top['tconst'])

    def _read_tsv(self, path: str) -> pl.DataFrame:
        return pl.read_csv(
            path, 
            separator='\t', 
            null_values="\\N", 
            quote_char=None, 
            ignore_errors=True
        )
    
    def _enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col('Title').is_in(self.watched).alias('watched'),
            pl.col('Title').is_in(self.favourites).alias('favourites')
        ])


if __name__=="__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    client = OMDbClient(os.environ['API_KEY'])
    Extractor(client).run()
