import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Vary preprocessing and similarity functions


class TfIdf:

    def __init__(self):
        self.context_matrix = None
        self.vectorizer = TfidfVectorizer()

    def fit(self, contexts):
        self.context_matrix = self.vectorizer.fit_transform(contexts)
    
    def predict(self, question):
        y = self.vectorizer.transform([question])
        distances = np.dot(self.context_matrix, y.transpose())
        return distances.argmax()

    def predict_many(self, questions):
        y = self.vectorizer.transform(questions)
        distances = np.dot(self.context_matrix, y.transpose())
        prediction = distances.argmax(axis=0)
        return np.array(prediction).reshape(-1)