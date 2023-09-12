import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List


def visualize_character_frequencies(timeseries: Dict[datetime, Dict[str, int]]):
    fig = go.Figure()

    for timestamp, character_frequencies in timeseries.items():
        characters = list(character_frequencies.keys())
        frequencies = list(character_frequencies.values())

        fig.add_trace(go.Bar(
            x=characters,
            y=frequencies,
            name=f'Character Frequencies ({timestamp})'
        ))

    fig.update_layout(title='Character Frequencies over Time', xaxis_title='Character', yaxis_title='Frequency')
    fig.show()


def visualize_word_frequencies(timeseries: Dict[datetime, Dict[str, int]]):
    fig = go.Figure()

    for timestamp, word_frequencies in timeseries.items():
        words = list(word_frequencies.keys())
        frequencies = list(word_frequencies.values())

        fig.add_trace(go.Bar(
            x=words,
            y=frequencies,
            name=f'Word Frequencies ({timestamp})'
        ))

    fig.update_layout(title='Word Frequencies over Time', xaxis_title='Word', yaxis_title='Frequency')
    fig.show()


def visualize_stopword_and_nonletter_frequencies(timeseries: Dict[datetime, Dict[str, int]]):
    fig = go.Figure()

    for timestamp, (stopword_frequencies, nonletter_frequencies) in timeseries.items():
        stopwords = list(stopword_frequencies.keys())
        stopwords_frequencies = list(stopword_frequencies.values())
        nonletter = list(nonletter_frequencies.keys())
        nonletter_frequencies = list(nonletter_frequencies.values())

        fig.add_trace(go.Bar(
            name=f'Stopwords ({timestamp})',
            x=stopwords,
            y=stopwords_frequencies
        ))

        fig.add_trace(go.Bar(
            name=f'Non-letter Characters ({timestamp})',
            x=nonletter,
            y=nonletter_frequencies
        ))

    fig.update_layout(
        title='Stopword and Non-letter Character Frequencies over Time',
        xaxis_title='Words/Characters',
        yaxis_title='Frequency',
        barmode='stack'
    )
    fig.show()
