import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from signature import Fingerprint
from umap import UMAP


def create_char_graph(text: str) -> nx.DiGraph:
    fingerprint = Fingerprint.from_text(text)
    char_graph = nx.DiGraph()
    normed_char_graph = nx.DiGraph()
    
    # Add nodes to the graph
    for character, frequency in fingerprint.NORMALIZED_CHARACTER_FREQUENCY.items():
        normed_char_graph.add_node(character, frequency=frequency)

    for character, frequency in fingerprint.CHARACTER_FREQUENCY.items():
        char_graph.add_node(character, frequency=frequency)
    
    return nx.tensor_product(normed_char_graph, char_graph)


def create_word_graph(text: str) -> nx.DiGraph:
    fingerprint = Fingerprint.from_text(text)
    word_graph = nx.DiGraph()
    normed_word_graph = nx.DiGraph()
    
    # Add nodes to the graph
    for word, frequency in fingerprint.WORD_FREQUENCY.items():
        word_graph.add_node(word, frequency=frequency)

    for word, frequency in fingerprint.NORMALIZED_WORD_FREQUENCY.items():
        normed_word_graph.add_node(word, frequency=frequency)
    # todo investigate adding negative frequencies
    return nx.tensor_product(normed_word_graph, word_graph)


def create_stopword_nonletter_graph(text: str) -> nx.DiGraph:
    fingerprint = Fingerprint.from_text(text)
    stopword_nonletter_graph = nx.DiGraph()

    # Add nodes for stopwords and non-letter characters
    for word, frequency in fingerprint.STOPWORD_FREQUENCY.items():
        stopword_nonletter_graph.add_node(word, frequency=frequency)

    for character, frequency in fingerprint.NONLETTER_FREQUENCY.items():
        stopword_nonletter_graph.add_node(character, frequency=frequency)

    return stopword_nonletter_graph


def convert_numpy_to_umap(graph: nx.DiGraph) -> np.ndarray:
    #TODO calculate adjacency matrix from graph
    umap_embedding = UMAP(n_components=2, metric='precomputed').fit_transform(adjacency_matrix)
    return umap_embedding

