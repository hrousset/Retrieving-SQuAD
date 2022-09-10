import pandas as pd
import numpy as np


def get_context(df):
    contexts = []
    for index, row in df.iterrows():
        for point in row['data']['paragraphs']:
            contexts.append(point['context'])
    return contexts

def get_questions(df):
    questions = []
    labels = []
    idx = 0
    for index, row in df.iterrows():
        for i in range(len(row['data']['paragraphs'])):
            for j in range(len(row['data']['paragraphs'][i]['qas'])):
                questions.append(row['data']['paragraphs'][i]['qas'][j]['question'])
                labels.append(idx)
            idx += 1
    return questions, labels

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path

        self.train_contexts = []
        self.train_questions = []
        self.train_labels = []
        self.test_contexts = []
        self.test_questions = []
        self.test_labels = []

    def load_train_contexts_and_questions(self, filename):
        path = self.data_path + "/" + filename
        df_train = pd.read_json(path)
        
        self.train_contexts = get_context(df_train)
        self.train_questions, self.train_labels = get_questions(df_train)

    def load_test_contexts_and_questions(self, filename):
        path = self.data_path + "/" + filename
        df_test = pd.read_json(path)
        
        self.test_contexts = get_context(df_test)
        self.test_questions, self.test_labels = get_questions(df_test)
