
from dataset.dataset import *
from util.config import *
from models.tf_idf_model import *


def main(contexts, question):

    model = TfIdf()
    model.fit(contexts)

    prediction = model.predict(question)
    
    print(contexts[prediction])
