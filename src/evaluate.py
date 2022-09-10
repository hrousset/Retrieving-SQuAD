import sys
sys.path.append("./")

from dataset.dataset import *
from util.config import *
from models.tf_idf_model import *


def main(config_file="configs/config.yml"):
    print("Evaluating train set")
    config = parse_config(config_file)

    dataset = Dataset(config["data_path"])
    dataset.load_test_contexts_and_questions(config['train_filename'])
    dataset.load_train_contexts_and_questions(config['test_filename'])
    print("Dataset Loaded")

    model = TfIdf()
    model.fit(dataset.train_contexts)

    accuracy = 0

    # prediction = model.predict_many(dataset.train_questions)


    for question in dataset.train_questions:

        prediction = model.predict(question[0])

        if prediction == question[1]:
            accuracy += 1
    
    accuracy /= len(dataset.train_questions)
    
    print("The accuracy on the train set is ", np.round(accuracy,2))

if __name__ == '__main__':
    main()