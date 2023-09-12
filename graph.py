import networkx as nx
import numpy as np
from signature import calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies, normalize_text
from umap import UMAP


def convert_graph_to_numpy(graph: nx.DiGraph) -> np.ndarray:
    node_indices = {node: index for index, node in enumerate(graph.nodes())}
    adjacency_matrix = nx.to_numpy_matrix(graph, nodelist=graph.nodes())
    return np.array(adjacency_matrix), node_indices


def convert_graphs_to_numpy(graphs: dict) -> dict:
    numpy_graphs = {}
    for graph_name, graph in graphs.items():
        numpy_graphs[graph_name] = convert_graph_to_numpy(graph)
    return numpy_graphs


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


def convert_numpy_to_umap(numpy_graph: tuple) -> np.ndarray:
    adjacency_matrix, _ = numpy_graph
    umap_embedding = UMAP(n_components=2, metric='precomputed').fit_transform(adjacency_matrix)
    return umap_embedding


# Example usage
text = "This is a sample text for graph creation."
normalized_text = normalize_text(text)
character_graph = create_character_graph(normalized_text)
word_graph = create_word_graph(normalized_text)
stopword_punctuation_graph = create_stopword_punctuation_graph(normalized_text)

# Convert graphs to numpy arrays
graphs = {
    'character_graph': character_graph,
    'word_graph': word_graph,
    'stopword_punctuation_graph': stopword_punctuation_graph
}
numpy_graphs = convert_graphs_to_numpy(graphs)

# Convert numpy arrays to UMAP embeddings
umap_embeddings = {graph_name: convert_numpy_to_umap(numpy_graph) for graph_name, numpy_graph in numpy_graphs.items()}

# Print the UMAP embeddings
for graph_name, umap_embedding in umap_embeddings.items():
    print(f"{graph_name} UMAP embedding:")
    print(umap_embedding)
    print()
