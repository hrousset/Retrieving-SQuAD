import sys
from sklearn.metrics import accuracy_score
sys.path.append("./")

from dataset.dataset import *
from util.config import *
from models.tf_idf_model import *
from models.bert_model import *
from util.metrics import mmr_score, accuracy_at_rank_score, document_accuracy_score


def main():
    config_file = sys.argv[1]
    config = parse_config(config_file)

    dataset = Dataset(config["data_path"])

    if config["dataset"]=='train':
        dataset.load_contexts_and_questions(config['train_filename'])
        print("Train dataset loaded :", len(dataset.contexts), "contexts and", len(dataset.labels), "questions")
    else:
        dataset.load_contexts_and_questions(config['test_filename'])
        print("Test dataset loaded", len(dataset.contexts), "contexts and", len(dataset.labels), "questions")

    chosen_model = config["model"]
    print("Loading", chosen_model, "model")

    if chosen_model == "TFIDF":
        model = TfIdf()
        model.fit(dataset.contexts)
        print("Infering questions")
        prediction = model.predict_many(dataset.questions)

    elif chosen_model == "BERT":
        model = BERT_model()
        
        if not config["load_context"]:
            print('Encoding contexts')
            model.fit(dataset.contexts)
            model.save_context_embedding(config["context_pickle_name"])
        else:
            print('Loading encoded contexts')
            model.load_context_embedding(config["context_pickle_name"])

        print("Infering questions")
        prediction = model.predict_many(dataset.questions, load=False, filename='train_questions')

    accuracy = accuracy_score(prediction[:,-1], dataset.labels)
    print("The accuracy on the train set is ", np.round(accuracy,2))

    accuracy_doc = document_accuracy_score(dataset.labels, np.array(prediction[:,-1]), dataset.paragraphe_to_document_map)
    print("The accuracy for finding the document is", accuracy_doc)
    
    mmr = mmr_score(dataset.labels, prediction)
    print("The MRR on the train set is", mmr)

    accuracy_at_rank = accuracy_at_rank_score(config["rank"], dataset.labels, prediction)
    print("The probability of being in first", config["rank"], "ranks is", accuracy_at_rank)

if __name__ == '__main__':
    main()
