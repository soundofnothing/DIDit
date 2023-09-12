import networkx as nx
from signature import calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies, normalize_text


def create_character_graph(text: str):
    # Generate character frequencies from the fingerprinted text
    character_frequencies = calculate_character_frequencies(text)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    for character, frequency in character_frequencies.items():
        graph.add_node(character, frequency=frequency)

    return graph


def create_word_graph(text: str):
    # Generate word frequencies from the fingerprinted text
    word_frequencies = calculate_word_frequencies(text)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    for word, frequency in word_frequencies.items():
        graph.add_node(word, frequency=frequency)

    return graph


def create_stopword_punctuation_graph(text: str):
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


# Example usage
text = "This is a sample text for graph creation."
normalized_text = normalize_text(text)
character_graph = create_character_graph(normalized_text)
word_graph = create_word_graph(normalized_text)
stopword_punctuation_graph = create_stopword_punctuation_graph(normalized_text)

# Print graph information
print("Character Graph nodes:", character_graph.nodes())
print("Word Graph nodes:", word_graph.nodes())
print("Stopword and Punctuation Graph nodes:", stopword_punctuation_graph.nodes())
