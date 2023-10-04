from collections import Counter
from typing import List, Dict, NamedTuple
import re
import math


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    #todo remove punctuation
    #todo perform stemming
    return text


def calculate_character_frequencies(text: str) -> Dict[str, int]:
    character_counts = Counter(text)
    return dict(character_counts)


def calculate_normalized_character_frequencies(text: str) -> Dict[str, int]:
    return calculate_character_frequencies(normalize_text(text))


def calculate_word_frequencies(text: str) -> Dict[str, int]:
    words = text.split()
    word_counts = Counter(words)
    return dict(word_counts)


def calculate_normalized_word_frequencies(text: str) -> Dict[str, int]:
    return calculate_word_frequencies(normalize_text(text))


def calculate_stopword_frequencies(text: str) -> Dict[str, int]:
    words = text.split()
    stopwords = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were']
    stopwords_counts = Counter(word for word in words if word in stopwords)
    return dict(stopwords_counts)

def calculate_nonletter_frequencies(text: str) -> Dict[str, int]:
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
    CHARACTER_FREQUENCY: Dict[str, int]
    NORMALIZED_CHARACTER_FREQUENCY: Dict[str, int]
    WORD_FREQUENCY: Dict[str, int]
    NORMALIZED_WORD_FREQUENCY: Dict[str, int]
    COSINE_SIMILARITY_CHAR: float
    COSINE_SIMILARITY_WORD: float
    STOPWORD_FREQUENCY: Dict[str, int]
    NONLETTER_FREQUENCY: Dict[str, int]

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

