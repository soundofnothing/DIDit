import dash
import dash_core_components as dcc
import dash_html_components as html
import graph
import timeline
import umap.plot
from dash.dependencies import Input, Output
from visualize import visualize_character_frequencies, visualize_word_frequencies, visualize_stopword_and_nonletter_frequencies


def process_textual_data(rss_feed: timeline.RSSFeed) -> dict:
    # Generate a stream of textual data from the RSS feed
    timeseries = timeline.generate_timeline_timeseries(rss_feed)

    # Initialize a dictionary to store the graphs
    graphs = {}

    # Process each entry in the timeseries
    for entry in timeseries:
        # Get the timestamped text from the entry
        timestamp, text = entry

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

    return graphs


def visualize_umap_embedding(embedding: any) -> dcc.Graph:
    # Example visualization using umap.plot.points
    figure = umap.plot.points(embedding)
    return dcc.Graph(figure=figure)


def visualize_word_graph(word_graph: any) -> dcc.Graph:
    # Example visualization using graph.create_word_graph
    figure = graph.create_word_graph(word_graph)
    return dcc.Graph(figure=figure)


if __name__ == "__main__":
    # Example usage
    rss_feed = timeline.RSSFeed.ACCOUNT1
    graphs = process_textual_data(rss_feed)

    # Create Dash app
    app = dash.Dash(__name__)

    # Set up Dash app layout
    app.layout = html.Div([
        dcc.Graph(id="character-frequencies", figure=visualize_character_frequencies(graphs)),
        dcc.Graph(id="word-frequencies", figure=visualize_word_frequencies(graphs)),
        dcc.Graph(id="stopword-nonletter-frequencies", figure=visualize_stopword_and_nonletter_frequencies(graphs)),
        dcc.Graph(id="umap-embedding", figure=visualize_umap_embedding(graph.convert_numpy_to_umap(graph.convert_graphs_to_numpy(graphs))))
    ])

    @app.callback(
        Output("word-frequencies", "figure"),
        [Input("character-frequencies", "hoverData")]
    )
    def update_word_graph(hoverData):
        if hoverData is not None:
            timestamp = hoverData["points"][0]["x"]
            word_graph = graphs.get(timestamp, {}).get("word", None)

            if word_graph is not None:
                return visualize_word_graph(word_graph)

        return None

    # Run the Dash app
    app.run_server(debug=True)
