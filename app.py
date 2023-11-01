import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from signature import Fingerprint
import streamlit as st
import requests
from bs4 import BeautifulSoup


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

    # Create a table for the fingerprint data
    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns]))
        ]
    )

    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Character Frequency', 'Word Frequency'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'table', 'colspan': 2}, None]],
        vertical_spacing=0.1
    )

    # Add traces to subplots
    fig.add_trace(char_scatter, row=1, col=1)
    fig.add_trace(word_scatter, row=1, col=2)
    fig.add_trace(table.data[0], row=2, col=1)

    # Update layout
    fig.update_layout(
        title_text="Character and Word Frequencies",
        height=800,
        width=1200
    )

    return fig, df

def get_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Failed to retrieve the URL: {e}")
        return None

def compare_fingerprints(inputs):
    figures = []
    data_tables = []
    for input_type, input_value in inputs:
        if input_type == "URL":
            input_value = get_text_from_url(input_value)
            if input_value is None:
                continue  # Skip to the next input if URL retrieval failed
        
        fingerprint = Fingerprint.from_text(input_value)
        fig, df = visualize_fingerprint_identity(fingerprint)
        figures.append(fig)
        data_tables.append(df)
    
    for i, (fig, df) in enumerate(zip(figures, data_tables)):
        st.write(f'### Text Snippet {i + 1}')
        st.plotly_chart(fig)
        st.write(df)

st.title("Authorship Inference")

num_inputs = st.number_input("Number of Inputs", min_value=1, value=1)

inputs = []
for i in range(num_inputs):
    input_type = st.selectbox(f"Input Type {i+1}", ["Text", "URL"], key=f"input_type_{i}")
    input_value = st.text_area(f"Enter Text or URL {i+1}", key=f"input_value_{i}")
    if input_value:  # Only add non-empty inputs
        inputs.append((input_type, input_value))

if st.button('Analyze'):
    if inputs:
        compare_fingerprints(inputs)
    else:
        st.error("Please enter at least one text snippet or URL.")