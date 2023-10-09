from collections import Counter
from typing import List, Dict, NamedTuple
import math
from textblob import TextBlob, Word, WordList


STOPWORDS = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were']


def normalize_text(text: str) -> str:
    text = TextBlob(text)
    # remove whitespace, covert to lowercase, attempt to correct spelling
    text = text.strip().lower().correct()
    # convert every word in a sentence to singular form
    words = [word.singularize() for word in text.words]
    # remove stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]
    # join the filtered words back into a single string
    return ' '.join(filtered_words)


def calculate_relative_character_frequencies(text: str) -> Dict[str, float]:
    # Calculate the total number of characters in the text
    total_characters = len(text)
    
    # Calculate character frequencies and divide by the total number of characters
    character_counts = Counter(text)
    character_frequencies = {char: count / total_characters for char, count in character_counts.items()}
    
    return character_frequencies


def calculate_normalized_character_frequencies(text: str) -> Dict[str, int]:
    return calculate_character_frequencies(normalize_text(text))


def calculate_relative_word_frequencies(text: str) -> Dict[str, float]:
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


def calculate_stopword_frequencies(text: str) -> Dict[str, float]:
    # Split the text into words
    words = text.split()
    
    # Define the stopwords list
    stopwords = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were']
    
    # Calculate the total number of stopwords
    total_stopwords = sum(1 for word in words if word in STOPWORDS)
    
    # Check if total_stopwords is zero to avoid division by zero
    if total_stopwords == 0:
        return {word: 0.0 for word in stopwords}
    
    # Calculate the stopwords counts and divide by the total number of stopwords
    stopwords_frequencies = {
        word: words.count(word) / total_stopwords
        for word in stopwords
    }
    
    return stopwords_frequencies


def calculate_nonletter_frequencies(text: str) -> Dict[str, float]:
    # Split the text into words
    words = text.split()
    
    # Calculate the total number of non-alphanumeric characters
    total_nonletters = sum(1 for char in words if not char.isalpha())
    
    # Calculate the nonletter counts and divide by the total number of non-alphanumeric characters
    nonletter_frequencies = {
        char: words.count(char) / total_nonletters
        for char in words if not char.isalpha()
    }
    
    return nonletter_frequencies


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
    char_freq = calculate_relative_character_frequencies(text)
    char_freq_normalized = calculate_normalized_character_frequencies(text)
    similarity = calculate_cosine_similarity(char_freq, char_freq_normalized)
    return similarity


def calculate_cosine_similarity_word(text: str) -> float:
    word_freq = calculate_relative_word_frequencies(text)
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
    
    # structural_deviation := COSINE_SIMILARITY_CHAR * character_delta * NONLETTER_FREQUENCY + COSINE_SIMILARITY_WORD * word_delta * STOPWORD_FREQUENCY
    STOPWORD_FREQUENCY: Dict[str, int]
    NONLETTER_FREQUENCY: Dict[str, int]

    # Calculate character_delta and word_delta as specified
    character_delta: Dict[str, int] = {}
    word_delta: Dict[str, int] = {}
    
    # Calculate structural_deviation as specified
    structural_deviation: float = 0.0

    # (character_delta, word_delta, and structural_deviation) is an "identity vector" for an author
    @property
    def identity_vector(self):
        return (self.character_delta, self.word_delta, self.structural_deviation)
        
    @classmethod
    def from_text(cls, text: str):
        # Calculate the required frequencies and similarities
        character_frequency = calculate_relative_character_frequencies(text)
        normalized_character_frequency = calculate_normalized_character_frequencies(text)
        word_frequency = calculate_relative_word_frequencies(text)
        normalized_word_frequency = calculate_normalized_word_frequencies(text)
        cosine_similarity_char = calculate_cosine_similarity_char(text)
        cosine_similarity_word = calculate_cosine_similarity_word(text)
        stopword_frequency = calculate_stopword_frequencies(text)
        nonletter_frequency = calculate_nonletter_frequencies(text)

        # Calculate character_delta and word_delta as specified
        character_delta = {}
        for char in character_frequency:
            if char in normalized_character_frequency:
                character_delta[char] = abs(character_frequency[char] - normalized_character_frequency[char])

        word_delta = {}
        for word in word_frequency:
            if word in normalized_word_frequency:
                word_delta[word] = abs(word_frequency[word] - normalized_word_frequency[word])

        # Calculate structural_deviation as specified
        structural_deviation = (cosine_similarity_char * sum(character_delta.values()) * sum(nonletter_frequency.values()) +
                                cosine_similarity_word * sum(word_delta.values()) * sum(stopword_frequency.values()))

        return cls(
            CHARACTER_FREQUENCY=character_frequency,
            NORMALIZED_CHARACTER_FREQUENCY=normalized_character_frequency,
            WORD_FREQUENCY=word_frequency,
            NORMALIZED_WORD_FREQUENCY=normalized_word_frequency,
            COSINE_SIMILARITY_CHAR=cosine_similarity_char,
            COSINE_SIMILARITY_WORD=cosine_similarity_word,
            STOPWORD_FREQUENCY=stopword_frequency,
            NONLETTER_FREQUENCY=nonletter_frequency,
            character_delta=character_delta,
            word_delta=word_delta,
            structural_deviation=structural_deviation
        )

def __str__(self):
    return str(self.identity_vector)

