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

def get_paragraphe_to_document_map(df):
    map = {}
    idx = 0
    for index, row in df.iterrows():
        for i in range(len(row['data']['paragraphs'])):
            for j in range(len(row['data']['paragraphs'][i]['qas'])):
                map[idx] = index
                idx += 1
    return map

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path

        self.contexts = []
        self.questions = []
        self.labels = []
        self.paragraphe_to_document_map = None

    def load_contexts_and_questions(self, filename):
        path = self.data_path + "/" + filename
        df = pd.read_json(path)
        
        self.contexts = get_context(df)
        self.questions, self.labels = get_questions(df)
        self.paragraphe_to_document_map = get_paragraphe_to_document_map(df)
