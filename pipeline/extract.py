import requests
import config as c
import polars as pl

from concurrent.futures import ThreadPoolExecutor

class Extractor:
    def __init__(self, api_key: str, request_main_universe: bool):
        self.api_key = api_key
        self.request_main_universe = request_main_universe
        self.base_url = "https://www.omdbapi.com"

        self.seen: pl.DataFrame | None = None
        self.seen_titles = []

    def run(self):
        self.seen = pl.read_csv(c.FILMS_CSV)
        self.seen_titles = self.seen['All'].drop_nulls().to_list()
        self.favourites = self.seen['Favourites'].drop_nulls().to_list()

        main_universe_df = self._retrieve_main_universe()
        main_universe_titles = set(main_universe_df['Title'])

        remaining = [i for i in self.seen_titles if i not in main_universe_titles]
        print(f"Manually retrieving {len(remaining)} movies")
        remaining_df = self._retrieve_remaining_titles(remaining).select(main_universe_df.columns)

        films = pl.concat([main_universe_df, remaining_df])
        films = self._enrich(films)
        films.write_parquet(c.EXTRACTED_PARQUET)
        print("Finished running extractor")

    def _retrieve_main_universe(self) -> pl.DataFrame:
        if self.request_main_universe:
            universe_title_ratings = self._read_tsv(c.RATINGS_TSV).sort('numVotes', descending=True).head(10000)
            with ThreadPoolExecutor(max_workers=10) as executor:
                dicts = list(executor.map(self._get_by_id, universe_title_ratings['tconst']))
                dicts = [r for r in dicts if r is not None]
            df = pl.from_dicts(dicts)
            df.write_parquet(c.MAIN_UNIVERSE_PARQUET)
        else:
            df = pl.read_parquet(c.MAIN_UNIVERSE_PARQUET)
        return df
    
    def _retrieve_remaining_titles(self, remaining: list[str]) -> pl.DataFrame:
        with ThreadPoolExecutor(max_workers=10) as executor:
            dicts = list(executor.map(self._get_by_title, remaining))
            dicts = [r for r in dicts if r is not None]
        return pl.from_dicts(dicts)

    def _enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .with_columns(pl.col('Title').is_in(self.seen_titles).alias('watched'))
            .with_columns(pl.col('Title').is_in(self.favourites).alias('favourites'))
        )
        return df
    
    def _get_data(self, title = None, id = None) -> dict | None:
        data = self._make_request(title, id)
        if not self._found_title(data):
            print(f"No result for '{title}'")
            return None
        return self._fix_ratings(data, title or id)

    def _get_by_title(self, title):
        return self._get_data(title=title)

    def _get_by_id(self, id):
        return self._get_data(id=id)

    def _make_request(self, title = None, id= None) -> dict:
        params = {'apikey': self.api_key}
        if title:
            params['t'] = title
        if id:
            params['i'] = id
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def _found_title(self, data: dict) -> bool:
        return data['Response'] == 'True'

    def _fix_ratings(self, data: dict, title_or_id) -> dict:
        data = data.copy()
        for d in data['Ratings']:
            source = d['Source']
            rating = d['Value']
            data[f'rating_{source.replace(" ", "_")}'] = rating
        del data['Ratings']
        data = {'Title_0': title_or_id or id, **data}
        return data
    
    def _read_tsv(self, path: str) -> pl.DataFrame:
        return pl.read_csv(
            path, 
            separator='\t', 
            null_values="\\N", 
            quote_char=None, 
            ignore_errors=True
        )
