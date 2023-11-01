import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from typing import List
from signature import Fingerprint
import plotly.figure_factory as ff
import plotly.subplots as sp
import plotly.graph_objects as go

# Define a function to pad character frequencies with zeros
def pad_char_frequencies(fp, char_labels):
    return [fp.NORMALIZED_CHARACTER_FREQUENCY.get(char, 0) for char in char_labels]


# Define the display function with word-wrapped titles
def display_fingerprints(fingerprints, titles=None, rows=1, cols=None, figsize=(10, 5), title_length=24):
    """
    Display multiple fingerprints in a grid of subplots with word-wrapped titles.

    Args:
        fingerprints (list): List of Fingerprint objects to display.
        titles (list): List of titles for each fingerprint (optional).
        rows (int): Number of rows in the grid (default is 1).
        cols (int): Number of columns in the grid (default is None, determined automatically).
        figsize (tuple): Figure size (width, height) in inches (default is (10, 5)).
        title_length (int): Maximum title length before word-wrapping (default is 24).
    """
    
    if cols is None:
        cols = len(fingerprints) // rows + (len(fingerprints) % rows > 0)

    plt.figure(figsize=figsize)

    for i, fingerprint in enumerate(fingerprints):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(display_fingerprints(fingerprint))
        plt.axis('off')

        # Word-wrap the title
        if titles and i < len(titles):
            wrapped_title = textwrap.fill(titles[i], title_length)
            plt.title(wrapped_title)

    plt.tight_layout()
    plt.show()


def render_heatmap(text_snippets: List[str]):
    fingerprints = [Fingerprint.from_text(text) for text in text_snippets]
    char_labels = list(fingerprints[0].NORMALIZED_CHARACTER_FREQUENCY.keys())
    char_freq_matrix = np.array([pad_char_frequencies(fp, char_labels) for fp in fingerprints])
    snippet_labels = [textwrap.wrap(text, 20) for text in text_snippets]

    # Convert Seaborn heatmap to Plotly figure
    fig = ff.create_annotated_heatmap(
        z=char_freq_matrix,
        x=char_labels,
        y=['\n'.join(label) for label in snippet_labels],
        colorscale='YlGnBu',
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title="Heatmap of Normalized Character Frequencies",
        xaxis_title="Character",
        yaxis_title="Text Snippet"
    )
    
    return fig


def visualize_fingerprint_identity(fingerprint):
    character_frequency = fingerprint.CHARACTER_FREQUENCY
    word_frequency = fingerprint.WORD_FREQUENCY
    
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Character Frequency', 'Word Frequency'))
    
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
    
    fig.add_trace(char_scatter, row=1, col=1)
    fig.add_trace(word_scatter, row=1, col=2)
    
    fig.update_layout(title_text="Character and Word Frequencies", height=600, width=1200)
    
    return fig
