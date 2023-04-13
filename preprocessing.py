import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download("stopwords")
nltk.download("wordnet")
wordnetLemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english"))


def lemmatizer(words):
    output = [wordnetLemmatizer.lemmatize(word) for word in words]
    return output


def countWords(words, bag):
    for word in words:
        if word not in bag:
            bag[word] = 0
        bag[word] += 1


def removeStopWord(words):
    output = []
    for word in words:
        if word in stopwords:
            continue
        output.append(word)
    return output


def labelWord(lines):
    sample_set = dict()
    count = 0
    for line in lines:
        for word in line:
            if word not in sample_set:
                sample_set[word] = count
                count += 1
    return sample_set


def save_json(map, filename):
    import json

    with open(filename, "w") as fd:
        json.dump(map, fd)


def generate_vector(line, mapping, vocab_size):
    line_output = np.zeros(vocab_size, dtype=np.float32)
    for word in line:
        line_output[mapping[word]] += 1
    return line_output


def generate_vectors(lines,mapping=None):
    if mapping==None: 
        mapping = labelWord(lines)
    result = []
    for line in lines:
        line_output = generate_vector(line,mapping,len(mapping)) 
        result.append(line_output)
    result = np.array(result)
    return result,mapping
