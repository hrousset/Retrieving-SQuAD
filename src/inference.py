import sys
sys.path.append("./")

from dataset.dataset import *
from util.config import parse_config
from models.tf_idf_model import *
from models.bert_model import *
from util.metrics import get_inferred_rank


def main():
    config_file = sys.argv[1]
    config = parse_config(config_file)

    dataset = Dataset(config["data_path"])

    if config["dataset"]=='train':
        dataset.load_contexts_and_questions(config['train_filename'])
        print("Train dataset loaded :", len(dataset.contexts), "contexts")
    else:
        dataset.load_contexts_and_questions(config['test_filename'])
        print("Test dataset loaded", len(dataset.contexts), "contexts")

    question = dataset.questions[config['question_idx']]
    label = dataset.labels[config['question_idx']]
    print("The chosen question is :", question)

    chosen_model = config["model"]
    print("Loading", chosen_model, "model")

    if chosen_model == "TFIDF":
        model = TfIdf()
        model.fit(dataset.contexts)
        print("Infering question")
        prediction = model.predict(question)

    elif chosen_model == "BERT":
        model = BERT_model()
        
        if config["load_context"]=='False':
            print('Encoding contexts')
            model.fit(dataset.contexts)
            model.save_context_embedding(config["context_pickle_name"])
        else:
            print('Loading encoded contexts')
            model.load_context_embedding(config["context_pickle_name"])

        print("Infering question")
        prediction = model.predict(question)
    
    print("The correct context is :", dataset.contexts[label])

    print("The predicted context is :", dataset.contexts[prediction[-1]])

    rank = get_inferred_rank(prediction, label)
    print("The correct context is ranked", rank)

    if dataset.paragraphe_to_document_map[label] == dataset.paragraphe_to_document_map[int(prediction[-1])]:
        print("The correct document is found.")
    else:
        print("The predicted document is not correct.")

if __name__ == '__main__':
    main()
