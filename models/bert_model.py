from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pickle


class BERT_model:

    def __init__(self):
        self.model = SentenceTransformer('msmarco-distilbert-base-v3')
        self.context_matrix = None

    def fit(self, contexts):
        self.context_matrix = self.model.encode(contexts)
    
    def predict(self, question):
        question_embedding = self.model.encode([question])
        distances = util.dot_score(question_embedding, self.context_matrix)
        return np.squeeze(distances.argsort(axis=1))
        

    def predict_many(self, questions, load=False, filename="question_encoding"):
        file_path = "models/saved_models/" + filename + ".pkl"
        if not load:
            question_embedding = self.model.encode(questions)
            with open(file_path, "wb") as fp:
                pickle.dump(question_embedding, fp)
        else:
            with open(file_path, "rb") as fp:
                question_embedding = pickle.load(fp)
        
        # distances = util.dot_score(question_embedding, self.context_matrix) #acc 0.47, doc acc 0.7, mrr 0.57, rank10 0.76
        distances = util.cos_sim(question_embedding, self.context_matrix) #acc 0.52, doc acc 0.73, mrr 0.61, rank10 0.79
        
        return distances.argsort(axis=1)


    def save_context_embedding(self, name):
        file_path = "models/saved_models/" + name + ".pkl"
        with open(file_path, "wb") as fp:
            pickle.dump(self.context_matrix, fp)


    def load_context_embedding(self, name):
        file_path = "models/saved_models/" + name + ".pkl"
        with open(file_path, "rb") as fp:
            self.context_matrix = pickle.load(fp)