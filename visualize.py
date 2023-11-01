import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from typing import List


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


# Define a function to pad character frequencies with zeros
def pad_char_frequencies(fp, char_labels):
    return [fp.NORMALIZED_CHARACTER_FREQUENCY.get(char, 0) for char in char_labels]


def render_heatmap(text_snippets: List[str]):
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


def render_heatmap(text_snippets: List[str]):
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
    for fingerprint in fingerprints:
        visualize_fingerprint_identity(fingerprint)


def visualize_fingerprint_identity(fingerprint):
    # Extract character and word frequencies
    character_frequency = fingerprint.CHARACTER_FREQUENCY
    word_frequency = fingerprint.WORD_FREQUENCY
    
    # Create a scatterplot for character frequencies
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(range(len(character_frequency)), list(character_frequency.values()), c='b', label='Character Frequency')
    plt.xticks(range(len(character_frequency)), list(character_frequency.keys()), rotation=90)
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.title('Character Frequency')
    
    # Create a scatterplot for word frequencies
    plt.subplot(122)
    plt.scatter(range(len(word_frequency)), list(word_frequency.values()), c='g', label='Word Frequency')
    plt.xticks(range(len(word_frequency)), list(word_frequency.keys()), rotation=90)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Word Frequency')
    
    plt.tight_layout()
    plt.show()
