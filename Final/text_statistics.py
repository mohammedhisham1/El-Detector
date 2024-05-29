import nltk
import pandas as pd


def total_arabic_word_count(text):
    # Remove punctuation marks and split the text into words
    words = text.split()
    # Initialize a counter for the total number of words
    total_count = 0

    # Iterate through each word
    for word in words:
        # Remove any non-Arabic characters (e.g., English words, numbers, etc.)
        cleaned_word = ''.join(c for c in word if '\u0600' <= c <= '\u06FF' or c == ' ')

        # Increment the total count if the cleaned word is not empty
        if cleaned_word.strip():
            total_count += 1

    return total_count


def total_arabic_character_count(text):
    # Initialize a counter for the total number of Arabic characters
    total_count = 0
    # Iterate through each character in the text
    for char in text:
        # Check if the character is Arabic
        if '\u0600' <= char <= '\u06FF':
            total_count += 1
    return total_count


def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def calculate_arabic_vocabulary(text):
    # Tokenize the text into words
    words = text.split()

    # Initialize a set to store unique Arabic words
    arabic_words = set()

    # Iterate through each word
    for word in words:
        # Remove any non-Arabic characters (e.g., English words, numbers, etc.)
        cleaned_word = ''.join(c for c in word if '\u0600' <= c <= '\u06FF')

        # Add the cleaned Arabic word to the set if it's not empty
        if cleaned_word.strip():
            arabic_words.add(cleaned_word)

    # Return the size of the Arabic vocabulary
    return len(arabic_words)


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def speech_speed_text(text):
    number_word = total_arabic_word_count(text)
    return convert_to_preferred_format((number_word / 125) * 100)


def read_speed_text(text):
    number_word = total_arabic_word_count(text)
    return convert_to_preferred_format((number_word / 200) * 100)


def show_text_statistics(texts):
    texts = texts.to_list()
    # Initialize an empty DataFrame with the appropriate column names
    columns = ['Text', 'Words', 'Characters', 'Sentence Count', 'Vocabulary',
               'Speech Speed', 'Read Speed']
    statistics = pd.DataFrame(columns=columns)
    statistics.style.set_properties(**{'text-align': 'left'})

    if isinstance(texts, list):
        for text in texts:
            # Collect the statistics for the current text
            data = [
                text,
                total_arabic_word_count(text),
                total_arabic_character_count(text),
                count_sentences(text),
                calculate_arabic_vocabulary(text),
                speech_speed_text(text),
                read_speed_text(text)
            ]
            # Append the data as a new row to the DataFrame
            statistics.loc[len(statistics)] = data
    else:
        # Collect the statistics for the single text
        data = [
            texts,
            total_arabic_word_count(texts),
            total_arabic_character_count(texts),
            count_sentences(texts),
            calculate_arabic_vocabulary(texts),
            speech_speed_text(texts),
            read_speed_text(texts)
        ]
        # Append the data as a new row to the DataFrame
        statistics.loc[len(statistics)] = data

    # Display the DataFrame
    return statistics
