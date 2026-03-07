import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def run_tfidf_plot_similarity(universe: pl.DataFrame, unseen: pl.DataFrame, favourites: pl.DataFrame) -> pl.DataFrame:

    universe_pd = universe.to_pandas()

    plots = universe_pd['plot'].fillna("")

    vectoriser = TfidfVectorizer(
        stop_words='english', # Ignore common words like 'the', 'is', 'of'
        max_features=8000, # Limit vocabulary to top 5000 words by TF-IDF weight
        ngram_range=(1,2)
    )

    # Each plot is converted into a vector of 8000 TF-IDF features
    tfidf_maxtrix = vectoriser.fit_transform(plots) 

    id_to_index = {id_: i for i, id_ in enumerate(universe['imdb_id'])}
    unseen_indices = [id_to_index[i] for i in unseen['imdb_id'] if i in id_to_index]
    favourite_indices = [id_to_index[i] for i in favourites['imdb_id'] if i in id_to_index]

    unseen_vectors = tfidf_maxtrix[unseen_indices] # type: ignore
    favourite_vectors = tfidf_maxtrix[favourite_indices] # type: ignore
    favourite_centroid = csr_matrix(favourite_vectors.mean(axis=0))

    similarity_matrix = cosine_similarity(unseen_vectors, favourite_centroid)
    similarity_scores = similarity_matrix.mean(axis=1)

    return pl.DataFrame({
        'imdb_id': unseen['imdb_id'],
        'tfidf_plot_similarity': similarity_scores
    })