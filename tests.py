import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
from datetime import datetime
from signature import calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies


def visualize_character_frequencies(text: str, timestamp: datetime = None):
    character_frequencies = calculate_character_frequencies(text)

    characters = list(character_frequencies.keys())
    frequencies = list(character_frequencies.values())

    fig = go.Figure(data=[go.Bar(x=characters, y=frequencies)])
    fig.update_layout(title='Character Frequencies', xaxis_title='Character', yaxis_title='Frequency')
    if timestamp:
        fig.update_layout(title=f'Character Frequencies ({timestamp})')
    fig.show()


def visualize_word_frequencies(text: str, timestamp: datetime = None):
    word_frequencies = calculate_word_frequencies(text)

    words = list(word_frequencies.keys())
    frequencies = list(word_frequencies.values())

    fig = px.bar(x=words, y=frequencies, labels={'x': 'Words', 'y': 'Frequency'})
    if timestamp:
        fig.update_layout(title=f'Word Frequencies ({timestamp})')
    fig.show()


def visualize_stopword_and_nonletter_frequencies(text: str, timestamp: datetime = None):
    stopword_frequencies, nonletter_frequencies = calculate_stopword_and_nonletter_frequencies(text)

    stopwords = list(stopword_frequencies.keys())
    stopwords_frequencies = list(stopword_frequencies.values())
    nonletter = list(nonletter_frequencies.keys())
    nonletter_frequencies = list(nonletter_frequencies.values())

    fig = go.Figure(data=[
        go.Bar(name='Stopwords', x=stopwords, y=stopwords_frequencies),
        go.Bar(name='Non-letter Characters', x=nonletter, y=nonletter_frequencies)
    ])
    fig.update_layout(title='Stopword and Non-letter Character Frequencies', xaxis_title='Words/Characters', yaxis_title='Frequency')
    if timestamp:
        fig.update_layout(title=f'Stopword and Non-letter Character Frequencies ({timestamp})')
    fig.show()
