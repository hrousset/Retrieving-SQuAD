import sys
from sklearn.metrics import accuracy_score
sys.path.append("./")

from dataset.dataset import *
from util.config import *
from models.tf_idf_model import *


def main():
    config_file = sys.argv[1]
    
    print("Evaluating train set")
    config = parse_config(config_file)

    dataset = Dataset(config["data_path"])
    dataset.load_test_contexts_and_questions(config['train_filename'])
    dataset.load_train_contexts_and_questions(config['test_filename'])
    print("Dataset Loaded")

    model = TfIdf()
    model.fit(dataset.train_contexts)

    prediction = model.predict_many(dataset.train_questions)
    accuracy = accuracy_score(prediction, dataset.train_labels)
    
    print("The accuracy on the train set is ", np.round(accuracy,2))

if __name__ == '__main__':
    main()
