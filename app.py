import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from visualize import render_heatmap, visualize_fingerprint_identity
from signature import Fingerprint
import requests

# Import dash_table.DataTable
import dash_table

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Text Analysis Dashboard"),
    
    # Input component for Dropbox URL
    dcc.Input(
        id='dropbox-url-input',
        type='text',
        placeholder="Enter Dropbox URL...",
        style={'width': '100%'},
    ),
    
    # Input component for entering text snippets
    dcc.Textarea(
        id='text-input',
        placeholder="Enter text snippets separated by newlines...",
        style={'width': '100%', 'height': '200px'},
    ),
    
    # Button to trigger analysis
    html.Button('Analyze', id='analyze-button'),
    
    # Heatmap to display normalized character frequencies
    dcc.Graph(id='heatmap'),
    
    # Identity visualization
    html.Div(id='identity-visualization'),

    # DataTable for browsing fingerprint data
    dash_table.DataTable(
        id='fingerprint-table',
        columns=[
            {"name": "Character Frequency", "id": "CHARACTER_FREQUENCY"},
            {"name": "Normalized Character Frequency", "id": "NORMALIZED_CHARACTER_FREQUENCY"},
            {"name": "Word Frequency", "id": "WORD_FREQUENCY"},
            {"name": "Normalized Word Frequency", "id": "NORMALIZED_WORD_FREQUENCY"},
            {"name": "Cosine Similarity (Char)", "id": "COSINE_SIMILARITY_CHAR"},
            {"name": "Cosine Similarity (Word)", "id": "COSINE_SIMILARITY_WORD"},
            {"name": "Stopword Frequency", "id": "STOPWORD_FREQUENCY"},
            {"name": "Nonletter Frequency", "id": "NONLETTER_FREQUENCY"},
            {"name": "Character Delta", "id": "character_delta"},
            {"name": "Word Delta", "id": "word_delta"},
            {"name": "Structural Deviation", "id": "structural_deviation"},
        ],
        style_table={'overflowX': 'scroll'},
        style_cell={'minWidth': '100px'},
        filter_action='native',
        sort_action='native',
        page_size=10,
    ),
])

# Define a function to fetch CSV data from Dropbox URL
def fetch_csv_data(dropbox_url):
    response = requests.get(dropbox_url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Define callback to prepopulate the textarea with CSV data
@app.callback(
    Output('text-input', 'value'),
    Input('dropbox-url-input', 'value')
)
def prepopulate_textarea(dropbox_url):
    if dropbox_url:
        csv_data = fetch_csv_data(dropbox_url)
        if csv_data:
            return csv_data
        else:
            return "Failed to fetch CSV data from Dropbox."
    else:
        return ""

# Define callback to update heatmap and DataTable
@app.callback(
    [Output('heatmap', 'figure'), Output('fingerprint-table', 'data')],
    Input('analyze-button', 'n_clicks'),
    Input('text-input', 'value')
)
def update_heatmap_and_table(n_clicks, text_input):
    if n_clicks and text_input:
        text_snippets = text_input.split('\n')
        fingerprints = [Fingerprint.from_text(text) for text in text_snippets]
        
        # Prepare data for the DataTable
        fingerprint_data = []
        for fingerprint in fingerprints:
            fingerprint_data.append({
                "CHARACTER_FREQUENCY": fingerprint.CHARACTER_FREQUENCY,
                "NORMALIZED_CHARACTER_FREQUENCY": fingerprint.NORMALIZED_CHARACTER_FREQUENCY,
                "WORD_FREQUENCY": fingerprint.WORD_FREQUENCY,
                "NORMALIZED_WORD_FREQUENCY": fingerprint.NORMALIZED_WORD_FREQUENCY,
                "COSINE_SIMILARITY_CHAR": fingerprint.COSINE_SIMILARITY_CHAR,
                "COSINE_SIMILARITY_WORD": fingerprint.COSINE_SIMILARITY_WORD,
                "STOPWORD_FREQUENCY": fingerprint.STOPWORD_FREQUENCY,
                "NONLETTER_FREQUENCY": fingerprint.NONLETTER_FREQUENCY,
                "character_delta": fingerprint.character_delta,
                "word_delta": fingerprint.word_delta,
                "structural_deviation": fingerprint.structural_deviation
            })
        
        return render_heatmap(text_snippets), fingerprint_data
    else:
        return {}, []

if __name__ == '__main__':
    app.run_server(debug=True)
