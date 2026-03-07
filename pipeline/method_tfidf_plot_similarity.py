import polars as pl
from polars.exceptions import ColumnNotFoundError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def run_tfidf_plot_similarity(
        universe: pl.DataFrame, 
        unseen: pl.DataFrame, 
        favourites: pl.DataFrame, 
        document_cols: list[str] = ['plot']
) -> pl.DataFrame:
    """Function to calculate a TF-IDF similarity score for each unseen movie.
    
    By default, it uses each movie's plot, but can use multiple features to create the document.
    """

    # Convert List(String) columns back to String columns
    for c in document_cols:
        if c not in universe.columns:
            raise ColumnNotFoundError(f"'{c}' not in DataFrame, check spelling.")
        
        if isinstance(universe[c].dtype, pl.List):
            universe = universe.with_columns(
                pl.col(c).list.join(' ').alias(c)
            )

    # Create list of movie 'documents'
    plots = (
        universe
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
    tfidf_matrix = vectoriser.fit_transform(plots) 

    id_to_index = {id_: i for i, id_ in enumerate(universe['imdb_id'])}
    unseen_indices = [id_to_index[i] for i in unseen['imdb_id'] if i in id_to_index]
    favourite_indices = [id_to_index[i] for i in favourites['imdb_id'] if i in id_to_index]

    unseen_vectors = tfidf_matrix[unseen_indices] # type: ignore
    favourite_vectors = tfidf_matrix[favourite_indices] # type: ignore
    favourite_centroid = csr_matrix(favourite_vectors.mean(axis=0))

    similarity_scores = cosine_similarity(unseen_vectors, favourite_centroid).flatten()

    return pl.DataFrame({
        'imdb_id': unseen['imdb_id'],
        'tfidf_document_similarity': similarity_scores
    })