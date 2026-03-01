import requests
import polars as pl

from concurrent.futures import ThreadPoolExecutor

class Extractor:
    def __init__(self, API_KEY: str, RETRIEVE_MAIN_UNIVERSE: bool, MAIN_UNIVERSE_EXTRACTED_PATH, FILMS_EXTRACTED_PATH):
        self.API_KEY = API_KEY
        self.RETRIEVE_MAIN_UNIVERSE = RETRIEVE_MAIN_UNIVERSE
        self.MAIN_UNIVERSE_EXTRACTED_PATH = MAIN_UNIVERSE_EXTRACTED_PATH
        self.EXTRACTED_DATA_PATH = FILMS_EXTRACTED_PATH
        self.BASE_URL = 'https://www.omdbapi.com'

        self.seen = None
        self.seen_titles = None

    def run(self):
        self.seen = pl.read_csv('data/films.csv')
        self.seen_titles = self.seen.select(pl.col('All')).to_series().to_list()

        main_universe_df = self._retrieve_main_universe()
        main_universe_titles = set(main_universe_df['Title'])

        remaining = [i for i in self.seen_titles if i not in main_universe_titles]
        remaining_df = self._retrieve_remaining_titles(remaining).select(main_universe_df.columns)

        films = pl.concat([main_universe_df, remaining_df])
        films = self._enrich(films, self.seen)
        films.write_parquet(self.EXTRACTED_DATA_PATH)
        print("Finished running extractor")

    def _retrieve_main_universe(self) -> pl.DataFrame:
        if self.RETRIEVE_MAIN_UNIVERSE:
            universe_title_ratings = self._read_tsv(path='data/title.ratings.tsv').sort('numVotes', descending=True).head(5000)
            with ThreadPoolExecutor(max_workers=10) as executor:
                main_universe_dicts = list(executor.map(self._get_by_id, universe_title_ratings['tconst']))
            df = pl.from_dicts(main_universe_dicts)
            df.write_parquet('data/main_universe_extracted.parquet')
        else:
            df = pl.read_parquet('data/main_universe_extracted.parquet')
        return df
    
    def _retrieve_remaining_titles(self, remaining: list[str]) -> pl.DataFrame:
        with ThreadPoolExecutor(max_workers=10) as executor:
            dicts = list(executor.map(self._get_by_title, remaining))
            dicts = [r for r in dicts if r is not None]
        return pl.from_dicts(dicts)

    def _enrich(self, df: pl.DataFrame, seen: pl.DataFrame) -> pl.DataFrame:
        forgotten = set(seen['Forgotten'].drop_nulls())
        great = set(seen['Great'].drop_nulls())
        favourites = set(seen['Amazing'].drop_nulls())

        df = df.with_columns(
            pl.when(pl.col('Title').is_in(self.seen_titles)).then(pl.lit(True)).otherwise(pl.lit(False)).alias('watched')
        )
        df = (
            df
            .with_columns(
                pl.when(pl.col('Title').is_in(forgotten) & pl.col('watched')).then(None)
                .when(pl.col('Title').is_in(great) & pl.col('watched')).then(pl.lit(2))
                .when(pl.col('Title').is_in(favourites) & pl.col('watched')).then(pl.lit(3))
                .when(~pl.col('watched')).then(None)
                .otherwise(pl.lit(1))
                .alias('rating_me')
            )
        )
        return df
    
    def _get_data(self, title = None, id = None) -> dict | None:
        data = self._make_request(title, id)
        if not self._found_title(data):
            print(f"No result for '{title}'")
            return None
        return self._fix_ratings(data, title, id)

    def _get_by_title(self, title):
        return self._get_data(title=title)

    def _get_by_id(self, id):
        return self._get_data(id=id)

    def _make_request(self, title = None, id= None) -> dict:
        params = {'apikey': self.API_KEY}
        if title:
            params['t'] = title
        if id:
            params['i'] = id
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _found_title(self, data: dict) -> bool:
        return data['Response'] == 'True'

    def _fix_ratings(self, data: dict, title: str, id: str) -> dict:
        data = data.copy()
        for d in data['Ratings']:
            source = d['Source']
            rating = d['Value']
            data[f'rating_{source.replace(" ", "_")}'] = rating
        del data['Ratings']
        data = {'Title_0': title or id, **data}
        return data
    
    def _read_tsv(self, path: str) -> pl.DataFrame:
        return pl.read_csv(
            path, 
            separator='\t', 
            null_values="\\N", 
            quote_char=None, 
            ignore_errors=True
        )
