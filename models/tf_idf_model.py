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
        self.vectorizer = TfidfVectorizer()
        # self.vectorizer = TfidfVectorizer(stop_words='english')
        # self.vectorizer = TfidfVectorizer(tokenizer=tokenize)
        # self.vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

    def fit(self, contexts):
        self.context_matrix = self.vectorizer.fit_transform(contexts)
    
    def predict(self, question):
        y = self.vectorizer.transform([question])
        distances = cosine_similarity(y, self.context_matrix)
        return distances.argmax()

    def predict_many(self, questions):
        y = self.vectorizer.transform(questions)
        
        # distances = np.dot(y, self.context_matrix)
        distances = cosine_similarity(y, self.context_matrix)

        prediction = distances.argmax(axis=1)
        return np.array(prediction).reshape(-1)