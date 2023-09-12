from signature import normalize_text, calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies, chunk_text
from datetime import datetime


# Test normalize_text function
def test_normalize_text():
    text = "Hello, World!"
    expected_output = "hello world"
    assert normalize_text(text) == expected_output


# Test calculate_character_frequencies function
def test_calculate_character_frequencies():
    text = "Hello, World!"
    expected_output = {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1}
    assert calculate_character_frequencies(text) == expected_output


# Test calculate_word_frequencies function
def test_calculate_word_frequencies():
    text = "Hello, World!"
    expected_output = {'hello': 1, 'world': 1}
    assert calculate_word_frequencies(text) == expected_output


# Test calculate_stopword_and_nonletter_frequencies function
def test_calculate_stopword_and_nonletter_frequencies():
    text = "Hello, World! This is a text with stopwords."
    expected_stopword_frequencies = {'is': 1, 'a': 1, 'this': 1, 'with': 1}
    expected_nonletter_frequencies = {',': 1, '!': 1, '.': 1}
    stopword_frequencies, nonletter_frequencies = calculate_stopword_and_nonletter_frequencies(text)
    assert stopword_frequencies == expected_stopword_frequencies
    assert nonletter_frequencies == expected_nonletter_frequencies


# Test chunk_text function
def test_chunk_text():
    data = ["This is the first text.", "This is the second text."]
    num_tokens = 3
    expected_output = ['This is the', 'first text.', 'This is the', 'second text.']
    assert chunk_text(data, num_tokens) == expected_output


# Run the test functions
test_normalize_text()
test_calculate_character_frequencies()
test_calculate_word_frequencies()
test_calculate_stopword_and_nonletter_frequencies()
test_chunk_text()
