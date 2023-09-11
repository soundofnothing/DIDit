import unittest
from signature import normalize_text, calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies

class TestSignature(unittest.TestCase):

    def test_normalize_text(self):
        text = "Hello, world!"
        expected_output = "hello world"
        self.assertEqual(normalize_text(text), expected_output)

    def test_calculate_character_frequencies(self):
        text = "hello world"
        expected_output = {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1}
        self.assertEqual(calculate_character_frequencies(text), expected_output)

    def test_calculate_word_frequencies(self):
        text = "hello world hello"
        expected_output = {'hello': 2, 'world': 1}
        self.assertEqual(calculate_word_frequencies(text), expected_output)

    def test_calculate_stopword_and_nonletter_frequencies(self):
        text = "Hello, world! This is a test text. It contains some stopwords like the, is, and the punctuation marks."
        expected_stopword_output = {'is': 1, 'the': 2, 'like': 1}
        expected_nonletter_output = {',': 1, '!': 1, '.': 2}
        stopwords, nonletter = calculate_stopword_and_nonletter_frequencies(text)
        self.assertEqual(stopwords, expected_stopword_output)
        self.assertEqual(nonletter, expected_nonletter_output)

if __name__ == '__main__':
    unittest.main()
