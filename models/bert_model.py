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
        question_embedding = self.model.encode(question)
        distances = util.dot_score(question_embedding, self.context_matrix)
        print(distances)
        return int(torch.argmax(distances))
        

    def predict_many(self, questions):
        # question_embedding = self.model.encode(questions)
        file_path = "models/saved_models/train_questions.pkl"
        # with open(file_path, "wb") as fp:
        #     pickle.dump(question_embedding, fp)
        with open(file_path, "rb") as fp:
            question_embedding = pickle.load(fp)
        
        # distances = util.dot_score(question_embedding, self.context_matrix)
        distances = util.cos_sim(question_embedding, self.context_matrix)
        return distances.argmax(axis=1)


    def save_context_embedding(self, name):
        file_path = "models/saved_models/" + name + ".pkl"
        with open(file_path, "wb") as fp:
            pickle.dump(self.context_matrix, fp)


    def load_context_embedding(self, name):
        file_path = "models/saved_models/" + name + ".pkl"
        with open(file_path, "rb") as fp:
            self.context_matrix = pickle.load(fp)