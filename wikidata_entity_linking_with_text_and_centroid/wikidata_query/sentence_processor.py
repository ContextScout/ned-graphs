import numpy as np
import networkx as nx

from wikidata_query.utils import add_triplets_to_graph_bw
from wikidata_query.utils import get_vectors_from_nodes_in_graph
from wikidata_query.utils import get_types_from_nodes_in_graph

#_cutoff = 200

_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def get_bw_graph(triplets):
#    triplets = triplets[:_cutoff]
    g_bw = nx.DiGraph()
    add_triplets_to_graph_bw(g_bw, triplets)
    return g_bw


def get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, model):
    g_bw = get_bw_graph(triplets)

    vectors = get_vectors_from_nodes_in_graph(g_bw, model)
    node_types = get_types_from_nodes_in_graph(g_bw)
    nodelist = g_bw.nodes()
    nodelist = sorted(nodelist, key=lambda x: x.split('|')[0] != central_item.lower())
    A_bw = np.array(nx.to_numpy_matrix(g_bw, nodelist=nodelist))
    return {'A_bw': A_bw,
            'vectors': vectors,
            'types': node_types}
