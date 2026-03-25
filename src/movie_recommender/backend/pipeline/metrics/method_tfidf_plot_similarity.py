import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, spmatrix


def run_tfidf_plot_similarity(
        universe: pl.DataFrame, 
        unseen: pl.DataFrame,
        favourites: pl.DataFrame,
        tfidf_matrix: spmatrix
) -> pl.DataFrame:
    """Function to calculate a TF-IDF similarity score for each unseen movie.
    
    By default, it uses each movie's plot, but can use multiple features to create the document.
    """

    id_to_index = {id_: i for i, id_ in enumerate(universe['imdb_id'])}
    unseen_indices = [id_to_index[i] for i in unseen['imdb_id'] if i in id_to_index]
    favourite_indices = [id_to_index[i] for i in favourites['imdb_id'] if i in id_to_index]

    unseen_vectors = tfidf_matrix[unseen_indices] # type: ignore
    favourite_vectors = tfidf_matrix[favourite_indices] # type: ignore
    favourite_centroid = csr_matrix(favourite_vectors.mean(axis=0))

    similarity_scores = cosine_similarity(unseen_vectors, favourite_centroid).flatten()
    similarity_scores_rounded = np.round(similarity_scores, 3)

    return pl.DataFrame({
        'imdb_id': unseen['imdb_id'],
        'tfidf_document_similarity': similarity_scores_rounded
    })