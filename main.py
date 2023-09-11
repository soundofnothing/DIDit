from signature import normalize_text, calculate_character_frequencies, calculate_word_frequencies, calculate_stopword_and_nonletter_frequencies

def main():
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

if __name__ == '__main__':
    main()
