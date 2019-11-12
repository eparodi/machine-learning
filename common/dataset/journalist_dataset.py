import codecs
import os.path
import pandas as pd
import re

from common.dataset.dataset import Dataset

dataset_folder_path = "../datasets/journalists"

def get_normalized_words(text):
    pattern = re.compile('[\W_]+')
    alphanumeric_text = pattern.sub(' ', text)
    words = alphanumeric_text.split(" ")
    return words

def average_sentence_length(text):
    sentences = re.split("(?<=[.!?])\s+", text)
    words_number = 0
    for sentence in sentences:
        words = sentence.split(" ")
        words_number += len(words)
    return words_number / len(sentences)

def different_used_words(text):
    words = get_normalized_words(text)
    different_words = list(dict.fromkeys(words))
    return len(different_words)
    
def coordinant_conjunctions_number(text):
    COORDINANTS = ["ni", "y", "o", "o bien", "pero aunque", "no obstante", "sin embargo",
        "sino", "por el contrario"]
    coordinant_number = 0
    for coordinant in COORDINANTS:
        coordinant_number += text.count(coordinant)
    return coordinant_number

def determinant_articles_frequency(text):
    ARTICLES = ["la", "el", "los", "las"]
    words = get_normalized_words(text)
    ret = 0
    for word in words:
        if word in ARTICLES:
            ret += 1
    return ret / len(words)

def indeterminant_articles_frequency(text):
    ARTICLES = ["una", "un", "unos", "unas"]
    words = get_normalized_words(text)
    ret = 0
    for word in words:
        if word in ARTICLES:
            ret += 1
    return ret / len(words)

def mente_adverbs_quantity(text):
    words = get_normalized_words(text)
    ret = 0
    for word in words:
        if word.endswith('mente'):
            ret += 1
    return ret

def relative_frequency_of_most_n_used_words(text, n):
    words = get_normalized_words(text)
    different_words = {word: words.count(word) / len(words) for word in words}
    appearance = list(different_words.values())
    appearance.sort(reverse=True)
    appearance = appearance[0:n]
    return sum(appearance)

def create_journalist_dataset():
    texts = []
    for folder in os.listdir(dataset_folder_path):
        folder_path = os.path.join(dataset_folder_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for files in os.listdir(folder_path):
            file_path = os.path.join(dataset_folder_path, folder, files)
            if not os.path.isfile(file_path):
                continue
            with codecs.open(file_path, "r", "ISO-8859-1") as content_file:
                text = content_file.read()
                text = text.lower()
                text = {
                    "journalist": folder,
                    "text": text,
                    "average_sentence_length": average_sentence_length(text),
                    "vocabulary_extension": different_used_words(text),
                    "coordinant_numbers": coordinant_conjunctions_number(text),
                    "indeterminant": indeterminant_articles_frequency(text),
                    "determinant": determinant_articles_frequency(text),
                    "mente_adverbs": mente_adverbs_quantity(text),
                    "most_used_words": relative_frequency_of_most_n_used_words(text, 5),
                }
                texts.append(text)

    df = pd.DataFrame(texts, columns=texts[0].keys())
    normalize_attrs = ["average_sentence_length",
                       "vocabulary_extension", "mente_adverbs", "coordinant_numbers"]
    for attr in normalize_attrs:
        df[attr] -= df[attr].min()
        df[attr] /= df[attr].max()
    return Dataset.build_dataset_from_rows(rows=df, clazz_attr="journalist", blacklisted_attrs="text")
