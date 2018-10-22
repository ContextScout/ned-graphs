import logging

import nltk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gensim import utils

_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]

_logger = logging.getLogger(__name__)


def get_words(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(utils.to_unicode(text))
    return words


def capitalize(word):
    return word[0].upper() + word[1:]


def low_case(word):
    return word[0].lower() + word[1:]


def infer_vector_from_word(model, word):
    vector = np.zeros(300)
    try:
        vector = model[word]
    except:
        try:
            vector = model[capitalize(word)]
        except:
            try:
                vector = model[low_case(word)]
            except:
                pass
    return vector


def infer_vector_from_doc(model, text):
    words = get_words(text)
    vector = np.zeros(300)
    for word in words:
        vector += infer_vector_from_word(model, word)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def get_vectors_from_nodes_in_graph(g, model):
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        text = node.replace('_', ' ')
        text = text.split('|')[0]
        vectors.append(infer_vector_from_doc(model, text))
    return np.array(vectors)


def get_types_from_nodes_in_graph(g):
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        texts = node.split('|')
        vector = np.zeros(3)
        if 'NODE' in texts:
            vector[0] = 1.
        if 'EDGE' in texts:
            vector[1] = 1.
        vectors.append(vector)
    return np.array(vectors)


def get_edge_name_with_signature(node_str):
    node_str = node_str.split('|')[0].lower()
    node_str += '|EDGE'
    return node_str


def get_node_name_with_signature(node_str):
    node_str = node_str.split('|')[0].lower()
    node_str += '|NODE'
    return node_str


def add_triplets_to_graph_bw(g, triplets):
    for n1, r, n2 in triplets:
        clean_n1 = get_node_name_with_signature(n1)
        clean_n2 = get_node_name_with_signature(n2)
        clean_r = get_edge_name_with_signature(r)
        g.add_node(clean_n1)
        g.add_node(clean_n2)
        g.add_node(clean_r)
        g.add_edge(clean_n2, clean_r, **{'label': 'to_relation'})
        g.add_edge(clean_r, clean_n1, **{'label': 'to_node'})
    return g


def add_triplets_to_graph_bw(g, triplets):
    for n1, r, n2 in triplets:
        clean_n1 = get_node_name_with_signature(n1)
        clean_n2 = get_node_name_with_signature(n2)
        clean_r = get_node_name_with_signature(r)
        g.add_node(clean_n1)
        g.add_node(clean_n2)
        g.add_node(clean_r)
        g.add_edge(clean_n2, clean_r, **{'label': 'to_relation'})
        g.add_edge(clean_r, clean_n1, **{'label': 'to_node'})
    return g


def plot_graph(g):
    layout = nx.shell_layout(g)
    nx.draw_networkx(g, pos=layout)
    nx.draw_networkx_edge_labels(g, pos=layout)
    plt.show()


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        seq_length = len(item['graph']['vectors'])
        question_length = len(item['question_vectors'])
        try:
            size_to_data_dict[(seq_length, question_length)].append(item)
        except:
            size_to_data_dict[(seq_length, question_length)] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets
