import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from signature import calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies, normalize_text
from umap import UMAP


def create_graph_from_count_matrix(count_matrix):
    # Create a graph from the count matrix
    graph = nx.from_scipy_sparse_matrix(count_matrix, create_using=nx.DiGraph)
    return graph


def create_graph_from_tfidf_matrix(tfidf_matrix):
    # Create a graph from the tf-idf matrix
    graph = nx.from_scipy_sparse_matrix(tfidf_matrix, create_using=nx.DiGraph)
    return graph


def create_character_graph(text: str) -> nx.DiGraph:
    # Generate character frequencies from the fingerprinted text
    character_frequencies = calculate_character_frequencies(text)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    for character, frequency in character_frequencies.items():
        graph.add_node(character, frequency=frequency)

    return graph


def create_word_graph(text: str) -> nx.DiGraph:
    # Generate word frequencies from the fingerprinted text
    word_frequencies = calculate_word_frequencies(text)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    for word, frequency in word_frequencies.items():
        graph.add_node(word, frequency=frequency)

    return graph


def create_stopword_punctuation_graph(text: str) -> nx.DiGraph:
    # Generate stopword and punctuation frequencies from the fingerprinted text
    stopword_frequencies, punctuation_frequencies = calculate_stopword_and_nonletter_frequencies(text)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add stopword nodes to the graph
    for stopword, frequency in stopword_frequencies.items():
        graph.add_node(stopword, frequency=frequency, type='stopword')

    # Add punctuation nodes to the graph
    for punctuation, frequency in punctuation_frequencies.items():
        graph.add_node(punctuation, frequency=frequency, type='punctuation')

    return graph


def convert_graph_to_numpy(graph: nx.DiGraph) -> np.ndarray:
    node_indices = {node: index for index, node in enumerate(graph.nodes())}
    adjacency_matrix = nx.to_numpy_matrix(graph, nodelist=graph.nodes())
    return np.array(adjacency_matrix), node_indices


def convert_graphs_to_numpy(graphs: dict) -> dict:
    numpy_graphs = {}
    for graph_name, graph in graphs.items():
        numpy_graphs[graph_name] = convert_graph_to_numpy(graph)
    return numpy_graphs


def convert_numpy_to_umap(numpy_graph: tuple) -> np.ndarray:
    adjacency_matrix, _ = numpy_graph
    umap_embedding = UMAP(n_components=2, metric='precomputed').fit_transform(adjacency_matrix)
    return umap_embedding
