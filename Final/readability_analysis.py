import re
import pandas as pd
from collections import Counter
import math


class LexicalFeatures:

    def __init__(self, texts):
        self.texts = texts

    def tokenize_words(self, text):
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if self.is_arabic(word)]

    def tokenize_sentences(self, text):
        sentences = re.split(r'[.!?]', text)
        return [sentence for sentence in sentences if sentence]

    def is_arabic(self, word):
        return any('\u0600' <= char <= '\u06FF' for char in word)

    def average_word_length(self, words):
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)

    def average_sentence_length_by_word(self, sentences):
        if not sentences:
            return 0
        return sum(len(self.tokenize_words(sentence)) for sentence in sentences) / len(sentences)

    def average_sentence_length_by_character(self, sentences):
        if not sentences:
            return 0
        return sum(len(sentence) for sentence in sentences) / len(sentences)

    def special_character_count(self, text):
        special_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
        return len(special_chars)

    def average_syllable_per_word(self, words):
        if not words:
            return 0
        return sum(self.count_syllables(word) for word in words) / len(words)

    def count_syllables(self, word):
        vowels = "aeiouAEIOU\u0627\u0648\u0649"
        return sum(1 for char in word if char in vowels)

    def functional_words_count(self, words):
        functional_words = {'في', 'من', 'إلى', 'على', 'أن', 'و', 'ب', 'ل', 'ك', 'التي', 'الذي', 'عن'}
        return sum(1 for word in words if word in functional_words)

    def punctuation_count(self, text):
        punctuation = re.findall(r'[.،!?؛:]', text)
        return len(punctuation)

    def compute_features(self):
        data = {
            "Average Word Length": [],
            "Average Sentence Length By Word": [],
            "Average Sentence Length By Character": [],
            "Special Character Count": [],
            "Average Syllable per Word": [],
            "Functional Words Count": [],
            "Punctuation Count": []
        }

        for text in self.texts:
            words = self.tokenize_words(text)
            sentences = self.tokenize_sentences(text)

            data["Average Word Length"].append(self.average_word_length(words))
            data["Average Sentence Length By Word"].append(self.average_sentence_length_by_word(sentences))
            data["Average Sentence Length By Character"].append(self.average_sentence_length_by_character(sentences))
            data["Special Character Count"].append(self.special_character_count(text))
            data["Average Syllable per Word"].append(self.average_syllable_per_word(words))
            data["Functional Words Count"].append(self.functional_words_count(words))
            data["Punctuation Count"].append(self.punctuation_count(text))

        return pd.DataFrame(data)


class VocabularyRichnessFeatures:

    def __init__(self, texts):
        self.texts = texts

    def tokenize_words(self, text):
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if self.is_arabic(word)]

    def is_arabic(self, word):
        return any('\u0600' <= char <= '\u06FF' for char in word)

    def hapax_legomena(self, words):
        freq = Counter(words)
        return sum(1 for word in freq if freq[word] == 1)

    def hapax_dislegomena(self, words):
        freq = Counter(words)
        return sum(1 for word in freq if freq[word] == 2)

    def honores_r_measure(self, words):
        N = len(words)
        V1 = self.hapax_legomena(words)
        return V1 / N

    def sichel_measure(self, words):
        N = len(words)
        V2 = self.hapax_dislegomena(words)
        return V2 / N

    def brunets_measure_w(self, words):
        N = len(words)+1
        V = len(set(words))
        return (V - 0.17) / math.log(N)

    def yules_characteristic_k(self, words):
        freq = Counter(words)
        N = len(words)
        M = sum(f ** 2 for f in freq.values())
        return 10000 * (M - N) / N ** 2

    def shannon_entropy(self, words):
        freq = Counter(words)
        N = len(words)
        return -sum((f / N) * math.log(f / N) for f in freq.values())

    def simpsons_index(self, words):
        freq = Counter(words)
        N = len(words)
        return sum((f / N) ** 2 for f in freq.values())

    def compute_features(self):
        data = {
            "Hapax Legomenon": [],
            "Hapax DisLegemena": [],
            "Honores R Measure": [],
            "Sichel’s Measure": [],
            "Brunet’s Measure W": [],
            "Yule’s Characteristic K": [],
            "Shannon Entropy": [],
            "Simpson’s Index": []
        }

        for text in self.texts:
            words = self.tokenize_words(text)

            data["Hapax Legomenon"].append(self.hapax_legomena(words))
            data["Hapax DisLegemena"].append(self.hapax_dislegomena(words))
            data["Honores R Measure"].append(self.honores_r_measure(words))
            data["Sichel’s Measure"].append(self.sichel_measure(words))
            data["Brunet’s Measure W"].append(self.brunets_measure_w(words))
            data["Yule’s Characteristic K"].append(self.yules_characteristic_k(words))
            data["Shannon Entropy"].append(self.shannon_entropy(words))
            data["Simpson’s Index"].append(self.simpsons_index(words))

        return pd.DataFrame(data)


class ReadabilityScores:

    def __init__(self, texts):
        self.texts = texts

    def tokenize_words(self, text):
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if self.is_arabic(word)]

    def tokenize_sentences(self, text):
        sentences = re.split(r'[.!؟]', text)
        return [sentence for sentence in sentences if sentence]

    def is_arabic(self, word):
        return any('\u0600' <= char <= '\u06FF' for char in word)

    def syllable_count(self, word):
        vowels = "aeiouAEIOU\u0627\u0648\u0649"
        return sum(1 for char in word if char in vowels)

    def flesch_reading_ease(self, words, sentences):
        word_count = len(words)
        sentence_count = len(sentences)
        syllable_count = sum(self.syllable_count(word) for word in words)

        if sentence_count == 0 or word_count == 0:
            return 0

        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count

        # Simplified formula adapted for Arabic
        return 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)

    def flesch_kincaid_grade_level(self, words, sentences):
        word_count = len(words)
        sentence_count = len(sentences)
        syllable_count = sum(self.syllable_count(word) for word in words)

        if sentence_count == 0 or word_count == 0:
            return 0

        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count

        # Simplified formula adapted for Arabic
        return (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59

    def gunning_fog_index(self, words, sentences):
        word_count = len(words)
        sentence_count = len(sentences)

        if sentence_count == 0 or word_count == 0:
            return 0

        complex_words = sum(1 for word in words if self.syllable_count(word) > 2)
        words_per_sentence = word_count / sentence_count

        return 0.4 * (words_per_sentence + 100 * (complex_words / word_count))

    def dale_chall_readability_formula(self, words, sentences):
        word_count = len(words)
        sentence_count = len(sentences)

        if sentence_count == 0 or word_count == 0:
            return 0

        difficult_words = sum(1 for word in words if not self.is_common_word(word))
        percentage_difficult_words = (difficult_words / word_count) * 100
        words_per_sentence = word_count / sentence_count

        return 0.1579 * percentage_difficult_words + 0.0496 * words_per_sentence

    def is_common_word(self, word):
        # This is a simple heuristic. A more accurate list of common Arabic words can be used.
        common_words = {'في', 'من', 'إلى', 'على', 'أن', 'و', 'ب', 'ل', 'ك', 'التي', 'الذي', 'عن'}
        return word in common_words

    def shannon_entropy(self, words):
        freq = Counter(words)
        N = len(words)
        return -sum((f / N) * math.log(f / N) for f in freq.values())

    def simpsons_index(self, words):
        freq = Counter(words)
        N = len(words)
        return sum((f / N) ** 2 for f in freq.values())

    def compute_features(self):
        data = {
            "Flesch Reading Ease": [],
            "Flesch-Kincaid Grade Level": [],
            "Gunning Fog Index": [],
            "Dale Chall Readability Formula": [],
            "Shannon Entropy": [],
            "Simpson's Index": []
        }

        for text in self.texts:
            words = self.tokenize_words(text)
            sentences = self.tokenize_sentences(text)

            data["Flesch Reading Ease"].append(self.flesch_reading_ease(words, sentences))
            data["Flesch-Kincaid Grade Level"].append(self.flesch_kincaid_grade_level(words, sentences))
            data["Gunning Fog Index"].append(self.gunning_fog_index(words, sentences))
            data["Dale Chall Readability Formula"].append(self.dale_chall_readability_formula(words, sentences))
            data["Shannon Entropy"].append(self.shannon_entropy(words))
            data["Simpson's Index"].append(self.simpsons_index(words))

        return pd.DataFrame(data)

