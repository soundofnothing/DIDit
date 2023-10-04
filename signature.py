from collections import Counter
from typing import List, Dict, NamedTuple
import re
import math
import numpy as np
from textblob import TextBlob, Word, WordList


def normalize_text(text: str) -> str:
    text = TextBlob(text)
    # remove whitespace, covert to lowercase, attempt to correct spelling
    text = text.strip().lower().correct()
    # convert every word in a sentence to singular form, this also should remove punctuation
    return ' '.join(word.singularize() for word in text.words)


def calculate_character_frequencies(text: str) -> Dict[str, float]:
    # Calculate the total number of characters in the text
    total_characters = len(text)
    
    # Calculate character frequencies and divide by the total number of characters
    character_counts = Counter(text)
    character_frequencies = {char: count / total_characters for char, count in character_counts.items()}
    
    return character_frequencies


def calculate_normalized_character_frequencies(text: str) -> Dict[str, int]:
    return calculate_character_frequencies(normalize_text(text))


def calculate_word_frequencies(text: str) -> Dict[str, float]:
    # Split the text into words
    words = text.split()
    
    # Calculate the total number of words in the text
    total_words = len(words)
    
    # Calculate word frequencies and divide by the total number of words
    word_counts = Counter(words)
    word_frequencies = {word: count / total_words for word, count in word_counts.items()}
    
    return word_frequencies


def calculate_normalized_word_frequencies(text: str) -> Dict[str, int]:
    return calculate_word_frequencies(normalize_text(text))


def calculate_stopword_frequencies(text: str) -> Dict[str, int]:
    # do NOT convert to a blob, the underlying word frequencies would be changed and we need it for a baseline
    words = text.split()
    stopwords = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were']
    stopwords_counts = Counter(word for word in words if word in stopwords)
    return dict(stopwords_counts)

def calculate_nonletter_frequencies(text: str) -> Dict[str, int]:
    # do NOT convert to a blob, the punctuation would be removed and we need it for a baseline
    words = text.split()
    nonletter_counts = Counter(char for char in words if not char.isalpha())
    return dict(nonletter_counts)


def calculate_cosine_similarity(vector1: Dict[str, int], vector2: Dict[str, int]) -> float:
    # Calculate the dot product of the two vectors
    dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)

    # Calculate the magnitude (Euclidean norm) of each vector
    magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))

    # Handle division by zero by returning 0 if either vector has zero magnitude
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


def calculate_cosine_similarity_char(text: str) -> float:
    char_freq = calculate_character_frequencies(text)
    char_freq_normalized = calculate_normalized_character_frequencies(text)
    similarity = calculate_cosine_similarity(char_freq, char_freq_normalized)
    return similarity


def calculate_cosine_similarity_word(text: str) -> float:
    word_freq = calculate_word_frequencies(text)
    word_freq_normalized = calculate_normalized_word_frequencies(text)
    similarity = calculate_cosine_similarity(word_freq, word_freq_normalized)
    return similarity


class Fingerprint(NamedTuple):
    # character_delta := abs(character_freq - normed_character_freq)
    CHARACTER_FREQUENCY: Dict[str, int]
    NORMALIZED_CHARACTER_FREQUENCY: Dict[str, int]
    
    # word_delta := abs(word_freq - normed_word_freq)
    WORD_FREQUENCY: Dict[str, int]
    NORMALIZED_WORD_FREQUENCY: Dict[str, int]
    
    # measures the degree of character and word alignment with normed frequencies
    COSINE_SIMILARITY_CHAR: float
    COSINE_SIMILARITY_WORD: float

    # structural_deviation := COSINE_SIMILARITY_CHAR * character_delta + COSINE_SIMILARITY_WORD * word_delta
    STOPWORD_FREQUENCY: Dict[str, int]
    NONLETTER_FREQUENCY: Dict[str, int]

    # (character_delta, word_delta, and structural_deviation) is an "identity vector" for an author
    
    @classmethod
    def from_text(cls, text: str):
        return cls(
            CHARACTER_FREQUENCY=calculate_character_frequencies(text),
            NORMALIZED_CHARACTER_FREQUENCY=calculate_normalized_character_frequencies(text),
            WORD_FREQUENCY=calculate_word_frequencies(text),
            NORMALIZED_WORD_FREQUENCY=calculate_normalized_word_frequencies(text),
            COSINE_SIMILARITY_CHAR=calculate_cosine_similarity_char(text),
            COSINE_SIMILARITY_WORD=calculate_cosine_similarity_word(text),
            STOPWORD_FREQUENCY=calculate_stopword_frequencies(text),
            NONLETTER_FREQUENCY=calculate_nonletter_frequencies(text)
        )


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
    import matplotlib.pyplot as plt
    import textwrap
    if cols is None:
        cols = len(fingerprints) // rows + (len(fingerprints) % rows > 0)

    plt.figure(figsize=figsize)

    for i, fingerprint in enumerate(fingerprints):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fingerprint_to_pixel(fingerprint))
        plt.axis('off')

        # Word-wrap the title
        if titles and i < len(titles):
            wrapped_title = textwrap.fill(titles[i], title_length)
            plt.title(wrapped_title)

    plt.tight_layout()
    plt.show()


# Define a function to pad character frequencies with zeros
def pad_char_frequencies(fp, char_labels):
    return [fp.NORMALIZED_CHARACTER_FREQUENCY.get(char, 0) for char in char_labels]


def render_heatmap(text_snippets: List[str]):
    import matplotlib.pyplot as plt
    import textwrap
    fingerprints = [Fingerprint.from_text(text) for text in text_snippets]

    # Get the character labels from the first fingerprint
    char_labels = list(fingerprints[0].NORMALIZED_CHARACTER_FREQUENCY.keys())

    # Pad character frequencies for all fingerprints
    char_freq_matrix = np.array([pad_char_frequencies(fp, char_labels) for fp in fingerprints])

    # Define the labels for the x-axis (characters) and y-axis (text snippets)
    snippet_labels = [textwrap.wrap(text, 20) for text in text_snippets]

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(char_freq_matrix, cmap="YlGnBu", xticklabels=char_labels, yticklabels=snippet_labels)

    # Customize the heatmap appearance (e.g., add labels, title, etc.)
    plt.xlabel("Character")
    plt.ylabel("Text Snippet")
    plt.title("Heatmap of Normalized Character Frequencies")

    # Show the heatmap
    plt.show()
