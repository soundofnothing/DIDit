import graph
import timeline
import umap.plot
from visualize import visualize_character_frequencies, visualize_word_frequencies, visualize_stopword_and_nonletter_frequencies
from typing import Dict, Any


def process_textual_data(rss_feed: timeline.RSSFeed) -> None:
    # Generate a stream of textual data from the RSS feed
    timeseries: Dict[timeline.Timestamp, str] = timeline.generate_timeline_timeseries(rss_feed)

    # Initialize a dictionary to store the graphs
    graphs: Dict[timeline.Timestamp, Dict[str, Any]] = {}

    # Process each entry in the timeseries
    for timestamp, text in timeseries.items():
        # Create graphs from the text
        character_graph = graph.create_character_graph(text)
        word_graph = graph.create_word_graph(text)
        stopword_punctuation_graph = graph.create_stopword_punctuation_graph(text)

        # Store the graphs in the dictionary under the corresponding timestamp
        graphs[timestamp] = {
            "character": character_graph,
            "word": word_graph,
            "stopword_punctuation": stopword_punctuation_graph
        }

    # Convert the graphs to numpy arrays
    numpy_graphs = graph.convert_graphs_to_numpy(graphs)

    # Perform topological data analysis on the numpy graphs
    for timestamp, numpy_graph in numpy_graphs.items():
        umap_embedding = graph.convert_numpy_to_umap(numpy_graph)

        # Example visualization using UMAP embedding
        visualize_umap_embedding(umap_embedding, timestamp)

    # Visualize the character frequencies over time
    visualize_character_frequencies(timeseries)

    # Visualize the word frequencies over time
    visualize_word_frequencies(timeseries)

    # Visualize the stopword and non-letter character frequencies over time
    visualize_stopword_and_nonletter_frequencies(timeseries)

    # Add any additional processing or analysis steps here


def visualize_umap_embedding(embedding: Any, timestamp: timeline.Timestamp) -> None:
    # Example visualization using umap.plot.points
    umap.plot.points(embedding, title=f"UMAP Embedding at {timestamp}")


if __name__ == "__main__":
    # Example usage
    rss_feed = timeline.RSSFeed.ACCOUNT1
    process_textual_data(rss_feed)
