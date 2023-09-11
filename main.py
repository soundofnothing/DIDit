from collections import Counter
import string
from typing import Dict, Tuple, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def normalize_text(text: str) -> str:
    # Replace punctuation with whitespace
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Initialize the Porter stemmer
    stemmer = PorterStemmer()

    # Get the English stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Perform word stemming and remove stopwords
    normalized_text = ""
    for token in tokens:
        # Perform stemming
        stemmed_word = stemmer.stem(token)

        # Check if the stemmed word is a stop word
        if stemmed_word not in stop_words:
            normalized_text += stemmed_word + " "

    return normalized_text.strip()


def calculate_character_frequencies(text: str) -> Dict[str, int]:
    # Normalize the text using the normalize_text function
    normalized_text = normalize_text(text)

    # Remove whitespaces from the normalized text
    normalized_text = normalized_text.replace(" ", "")

    # Calculate the character frequencies
    character_frequencies = Counter(normalized_text)

    return character_frequencies


def calculate_word_frequencies(text: str) -> Dict[str, int]:
    # Tokenize the normalized text into words
    words = word_tokenize(text)

    # Calculate the word frequencies
    word_frequencies = Counter(words)

    return word_frequencies


def calculate_stopword_and_nonletter_frequencies(text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    # Normalize the text using the normalize_text function
    normalized_text = normalize_text(text)

    # Tokenize the normalized text into words
    words = word_tokenize(normalized_text)

    # Get the English stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Calculate the frequencies of stopwords and non-letter characters
    stopword_frequencies = Counter([word for word in words if word in stop_words])
    nonletter_frequencies = Counter([char for char in normalized_text if not char.isalpha()])

    return stopword_frequencies, nonletter_frequencies


# Main program
original_text = input("Enter the original text: ")
normalized_text = normalize_text(original_text)

character_frequencies = calculate_character_frequencies(normalized_text)
print("Normalized Text Character Frequencies:")
for char, frequency in character_frequencies.items():
    print(char, ":", frequency)

word_frequencies = calculate_word_frequencies(normalized_text)
print("Word Frequencies:")
for word, frequency in word_frequencies.items():
    print(word, ":", frequency)

stopword_frequencies, nonletter_frequencies = calculate_stopword_and_nonletter_frequencies(original_text)
print("Stopword Frequencies:")
for stopword, frequency in stopword_frequencies.items():
    print(stopword, ":", frequency)

print("Non-letter Character Frequencies:")
for char, frequency in nonletter_frequencies.items():
    print(char, ":", frequency)
