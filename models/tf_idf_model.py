from array import array
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import string

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class TfIdf:

    def __init__(self):
        self.context_matrix = None
        self.vectorizer = TfidfVectorizer() #acc  0.46, acc doc 0.7, mrr 0.57, rank10 0.79 on train set
        # self.vectorizer = TfidfVectorizer(stop_words='english') #acc 0.44, acc doc 0.69, mrr 0.55, rank10 0.77 on train set
        # self.vectorizer = TfidfVectorizer(tokenizer=tokenize) #acc 0.44, doc acc 0.67, mrr 0.55, rank10 0.76 on train set
        # self.vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english') #acc 0.42, doc acc 0.66, mrr 0.53, rank10 0.74 on train set

    def fit(self, contexts):
        self.context_matrix = self.vectorizer.fit_transform(contexts)
    
    def predict(self, question):
        y = self.vectorizer.transform([question])
        distances = cosine_similarity(y, self.context_matrix)
        return np.squeeze(distances.argsort(axis=1))

    def predict_many(self, questions):
        y = self.vectorizer.transform(questions)
        # distances = (y @ self.context_matrix.T).toarray() #acc 0.46, doc acc 0.7, mrr 0.57, rank10 0.79
        distances = cosine_similarity(y, self.context_matrix)
        prediction = distances.argsort(axis=1)
        return prediction
