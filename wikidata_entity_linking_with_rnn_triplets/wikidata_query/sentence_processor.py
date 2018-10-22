import numpy as np

from wikidata_query.utils import infer_vector_from_doc

_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def create_vectors_from_triplets(triplets, model):
    return [np.concatenate([
        infer_vector_from_doc(model, item[0]),
        infer_vector_from_doc(model, item[1]),
        infer_vector_from_doc(model, item[2])])
        for item in triplets]


def get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, model):
    vectors = create_vectors_from_triplets(triplets, model)
    return {'vectors': vectors}
