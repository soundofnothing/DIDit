from signature import normalize_text, calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies, chunk_text

# Normalize text
text = "Hello, World!"
normalized_text = normalize_text(text)
print(normalized_text)

# Calculate character frequencies
character_frequencies = calculate_character_frequencies(text)
print(character_frequencies)

# Calculate word frequencies
word_frequencies = calculate_word_frequencies(text)
print(word_frequencies)

# Calculate stopword and nonletter frequencies
stopword_frequencies, nonletter_frequencies = calculate_stopword_and_nonletter_frequencies(text)
print(stopword_frequencies)
print(nonletter_frequencies)

# Chunk text
data = ["This is the first text.", "This is the second text."]
num_tokens = 3
chunks = chunk_text(data, num_tokens)
print(chunks)
