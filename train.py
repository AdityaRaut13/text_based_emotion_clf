from preprocessing import generate_vectors, countWords, lemmatizer
import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_set(lines):
    count = 0
    labels = dict()
    for label in lines:
        if label not in labels:
            labels[label] = count
            count += 1
    return labels


def test_sample(line, mapping):
    result = []
    for word in line:
        if word not in mapping:
            continue
        result.append(word)
    return result


def vocab_size(df, col):
    bag = dict()
    df[col].apply(lambda line: countWords(line, bag))
    return len(bag)


def generate_tfm(lines: pd.Series, mapping: dict):
    result = []
    for line in lines:
        line_output = np.zeros(len(mapping))
        for word in line:
            line_output[mapping[word]] = 1
        result.append(line_output)
    result = np.array(result)
    return result


def generate_tf_idf(lines: pd.Series, mapping: dict):
    tfm = generate_tfm(lines, mapping)
    idf = np.log((1 + tfm.shape[0]) / (tfm.sum(axis=0) + 1)) + 1
    tf_idf = tfm * idf
    distance = np.sqrt(np.sum(tf_idf**2, axis=1))
    for index, doc in enumerate(tf_idf):
        tf_idf[index] = doc / distance[index]
    return tf_idf


train = pd.read_csv("dataset/train.txt", sep=";", names=["text", "label"])
test = pd.read_csv("dataset/test.txt", sep=";", names=["text", "label"])

# subsituting words with changing meaning
train["words"] = train["text"].apply(lambda line: lemmatizer(line.split(" ")))

# converting to the count words for each document
from sklearn.feature_extraction.text import TfidfVectorizer

X, mapping = generate_vectors(train["words"])
# TrainX_tf_idf = TfidfVectorizer(train["words"])

TrainX_tf_idf = generate_tf_idf(train["words"], mapping)

# converting the text label into numbers
label = generate_set(train["label"])
train["Y"] = train["label"].apply(lambda word: label[word])


# subsituting words without changing meaning
# removing the words that are not in vocablary
test["words"] = test["text"].apply(
    lambda sentence: test_sample(lemmatizer(sentence.split(" ")), mapping)
)
print(" Test Vocab Size : ", vocab_size(test, "words"))
print(len(mapping))

# converting to the count words for each document
Test_X, _ = generate_vectors(test["words"], mapping)
TestX_tf_idf = generate_tf_idf(test["words"], mapping)


test["Y"] = test["label"].apply(lambda emotion: label[emotion])

print("Train Vocab size : ", vocab_size(train, "words"))


from sklearn.naive_bayes import MultinomialNB

nb_bg = MultinomialNB(force_alpha=True)
nb_bg.fit(X, train["Y"])
print("[ Naive Bayes ] bag of words : ", nb_bg.score(Test_X, test["Y"]))

nb_tf = MultinomialNB(force_alpha=True)
nb_tf.fit(TrainX_tf_idf, train["Y"])
print("[ Naive Bayes ] tf-idf : ", nb_tf.score(TestX_tf_idf, test["Y"]))

# from sklearn import svm
# 
# clf_bg = svm.SVC(verbose=True)
# clf_bg.fit(X, train["Y"])
# print("[ SVM ] bag of words : ", clf_bg.score(Test_X, test["Y"]))
# 
# clf_tf = svm.SVC()
# clf_tf.fit(TrainX_tf_idf, train["Y"])
# print("[ SVM ] tf-idf : ", clf_tf.score(TestX_tf_idf, test["Y"]))


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

gpc_bg=GaussianProcessClassifier(1.0 * RBF(1.0))
gpc_bg.fit(X,train["Y"])
print("[ GPC ] bag of words : ",gpc_bg.score(Test_X,test["Y"]))


gpc_tf=GaussianProcessClassifier(1.0 * RBF(1.0))
gpc_tf.fit(TrainX_tf_idf, train["Y"])
print("[ GPC ] tf-idf : ", gpc_tf.score(TestX_tf_idf, test["Y"]))
