# import dash
# import dash_table
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# from signature import Fingerprint

# # Initialize the Dash app
# app = dash.Dash(__name__)

# # Define the layout of the app
# app.layout = html.Div([
#     html.H1("Text Analysis Dashboard"),
    
#     # Input component for entering text snippets
#     dcc.Textarea(
#         id='text-input',
#         placeholder="Enter text snippets separated by newlines...",
#         style={'width': '100%', 'height': '200px'},
#     ),
    
#     # Button to trigger analysis
#     html.Button('Analyze', id='analyze-button'),
    
#     # DataTable for browsing fingerprint data
#     dash_table.DataTable(
#         id='fingerprint-table',
#         columns=[
#             {"name": i, "id": i} for i in ["Character Frequency", "Normalized Character Frequency", 
#                                             "Word Frequency", "Normalized Word Frequency", 
#                                             "Cosine Similarity (Char)", "Cosine Similarity (Word)",
#                                             "Stopword Frequency", "Nonletter Frequency",
#                                             "Character Delta", "Word Delta", "Structural Deviation"]
#         ],
#         style_table={'overflowX': 'scroll'},
#         style_cell={'minWidth': '100px'},
#         filter_action='native',
#         sort_action='native',
#         page_size=10,
#     ),
# ])

# # Define callback to update DataTable
# @app.callback(
#     Output('fingerprint-table', 'data'),
#     Input('analyze-button', 'n_clicks'),
#     Input('text-input', 'value')
# )
# def update_table(n_clicks, text_input):
#     if n_clicks and text_input:
#         text_snippets = text_input.split('\n')
#         fingerprints = [Fingerprint.from_text(text) for text in text_snippets]

#         # Prepare data for the DataTable
#         fingerprint_data = []
#         for fingerprint in fingerprints:
#             fingerprint_data.append({
#                 "Character Frequency": fingerprint.CHARACTER_FREQUENCY,
#                 "Normalized Character Frequency": fingerprint.NORMALIZED_CHARACTER_FREQUENCY,
#                 "Word Frequency": fingerprint.WORD_FREQUENCY,
#                 "Normalized Word Frequency": fingerprint.NORMALIZED_WORD_FREQUENCY,
#                 "Cosine Similarity (Char)": fingerprint.COSINE_SIMILARITY_CHAR,
#                 "Cosine Similarity (Word)": fingerprint.COSINE_SIMILARITY_WORD,
#                 "Stopword Frequency": fingerprint.STOPWORD_FREQUENCY,
#                 "Nonletter Frequency": fingerprint.NONLETTER_FREQUENCY,
#                 "Character Delta": fingerprint.character_delta,
#                 "Word Delta": fingerprint.word_delta,
#                 "Structural Deviation": fingerprint.structural_deviation
#             })
        
#         return fingerprint_data
#     else:
#         return []

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from signature import Fingerprint

def visualize_fingerprint_identity(fingerprint):
    # Extract the required data
    character_frequency = fingerprint.CHARACTER_FREQUENCY
    word_frequency = fingerprint.WORD_FREQUENCY

    # Create scatter plots for character and word frequencies
    char_scatter = go.Scatter(
        x=list(character_frequency.keys()),
        y=list(character_frequency.values()),
        mode='markers',
        name='Character Frequency'
    )

    word_scatter = go.Scatter(
        x=list(word_frequency.keys()),
        y=list(word_frequency.values()),
        mode='markers',
        name='Word Frequency'
    )

    # Prepare data for the table
    fingerprint_data = {
        "Character Frequency": fingerprint.CHARACTER_FREQUENCY,
        "Normalized Character Frequency": fingerprint.NORMALIZED_CHARACTER_FREQUENCY,
        "Word Frequency": fingerprint.WORD_FREQUENCY,
        "Normalized Word Frequency": fingerprint.NORMALIZED_WORD_FREQUENCY,
        "Cosine Similarity (Char)": fingerprint.COSINE_SIMILARITY_CHAR,
        "Cosine Similarity (Word)": fingerprint.COSINE_SIMILARITY_WORD,
        "Stopword Frequency": fingerprint.STOPWORD_FREQUENCY,
        "Nonletter Frequency": fingerprint.NONLETTER_FREQUENCY,
        "Character Delta": fingerprint.character_delta,
        "Word Delta": fingerprint.word_delta,
        "Structural Deviation": fingerprint.structural_deviation
    }

    # Convert data to DataFrame for display
    df = pd.DataFrame(fingerprint_data)

    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Character Frequency', 'Word Frequency', 'Fingerprint Data'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'table', 'colspan': 2}, None]],
        vertical_spacing=0.1
    )

    # Add traces to subplots
    fig.add_trace(char_scatter, row=1, col=1)
    fig.add_trace(word_scatter, row=1, col=2)
    fig.add_trace(
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns])
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title_text="Character and Word Frequencies",
        height=800,
        width=1200
    )

    return fig

# Example usage:
fingerprint = Fingerprint.from_text("Your text here")
fig = visualize_fingerprint_identity(fingerprint)
fig.show()
