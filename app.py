import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from signature import Fingerprint
import streamlit as st

def create_fingerprint(text):
    return Fingerprint.from_text(text)


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

def compare_fingerprints(texts):
    figures = []
    data_tables = []
    for text in texts:
        fingerprint = Fingerprint.from_text(text)
        fig, df = visualize_fingerprint_identity(fingerprint)
        figures.append(fig)
        data_tables.append(df)
    
    for i, (fig, df) in enumerate(zip(figures, data_tables)):
        st.write(f'### Text Snippet {i + 1}')
        st.plotly_chart(fig)
        st.write(df)
    
st.title("Authorship Inference")

# Let the user specify the number of text snippets they want to compare
num_texts = st.number_input("Number of Text Snippets", min_value=1, value=1)

# Dynamically create that number of text boxes
text_boxes = [st.text_area(f"Enter Text Snippet {i+1}") for i in range(num_texts)]

if st.button('Analyze'):
    # Now text_boxes is a list of strings containing the text from each text box
    texts_to_compare = [box for box in text_boxes if box]  # This filters out any empty text boxes
    if texts_to_compare:
        compare_fingerprints(texts_to_compare)
    else:
        st.error("Please enter at least one text snippet.")