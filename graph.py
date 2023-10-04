import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from signature import Fingerprint
from typing import Dict
from umap import UMAP


def create_frequency_graph(frequencies: Dict[str, int]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for datapoint, frequency in frequencies.items():
        graph.add_node(datapoint, frequency=frequency)
    return graph

def create_char_graph_embedding(text: str) -> nx.DiGraph:
    fingerprint = Fingerprint.from_text(text)
    char_graph = nx.DiGraph()
    normed_char_graph = nx.DiGraph()
    
    # Add nodes to the graph
    normed_char_graph = create_frequency_graph(fingerprint.NORMALIZED_CHARACTER_FREQUENCY.items())
    char_graph = create_frequency_graph(fingerprint.CHARACTER_FREQUENCY.items())
    
    return nx.tensor_product(normed_char_graph, nx.union(normed_char_graph, char_graph))


def create_word_graph_embedding(text: str) -> nx.DiGraph:
    fingerprint = Fingerprint.from_text(text)
    word_graph = nx.DiGraph()
    normed_word_graph = nx.DiGraph()
    
    # Add nodes to the graph
    normed_word_graph = create_frequency_graph(fingerprint.NORMALIZED_WORD_FREQUENCY.items())
    word_graph = create_frequency_graph(fingerprint.WORD_FREQUENCY.items())
    
    return nx.tensor_product(normed_word_graph, nx.union(normed_word_graph, word_graph))


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

