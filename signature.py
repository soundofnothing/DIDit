from collections import Counter
from typing import List, Dict
import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def calculate_character_frequencies(text: str) -> Dict[str, int]:
    text = normalize_text(text)
    character_counts = Counter(text)
    return dict(character_counts)


def calculate_word_frequencies(text: str) -> Dict[str, int]:
    text = normalize_text(text)
    words = text.split()
    word_counts = Counter(words)
    return dict(word_counts)


def calculate_stopword_and_nonletter_frequencies(text: str) -> Dict[str, int]:
    text = normalize_text(text)
    words = text.split()
    stopwords = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were']
    stopwords_counts = Counter(word for word in words if word in stopwords)
    nonletter_counts = Counter(char for char in text if not char.isalpha())
    return dict(stopwords_counts), dict(nonletter_counts)


def chunk_text(data: List[str], num_tokens: int) -> List[str]:
    chunks = []
    for text in data:
        tokens = text.split()
        num_chunks = len(tokens) // num_tokens
        text_chunks = [' '.join(tokens[i:i+num_tokens]) for i in range(0, len(tokens), num_tokens)]
        chunks.extend(text_chunks)
    return chunks
