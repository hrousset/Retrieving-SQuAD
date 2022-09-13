import numpy as np


def get_reciprocal_rank(row):
    label = row[-1]
    rank = np.where(row[0:len(row)-1]==label)[0]
    return 1/(len(row)-1-rank)

def get_rank(row):
    label = row[-1]
    rank = np.where(row[0:len(row)-1]==label)[0]
    return len(row)-1-rank

def document_accuracy_score(labels, prediction, map):
    accuracy_doc = 0
    for i in range(len(labels)):
        if map[labels[i]] == map[prediction[i]]:
            accuracy_doc += 1
    return np.round(accuracy_doc/len(labels), 2)

def mmr_score(labels, prediction):
    labels_exp = np.expand_dims(labels, axis=1)
    total_matrix = np.concatenate((prediction, labels_exp), axis=1)
    reciprocal_ranks = np.apply_along_axis(get_reciprocal_rank, axis=1, arr=total_matrix)
    return np.round(reciprocal_ranks.mean(), 2)


def accuracy_at_rank_score(rank, labels, prediction):
    labels_exp = np.expand_dims(labels, axis=1)
    total_matrix = np.concatenate((prediction, labels_exp), axis=1)
    ranks = np.apply_along_axis(get_rank, axis=1, arr=total_matrix)

    acc_rank = 0
    for i in ranks:
        if i <=rank:
            acc_rank += 1
    return np.round(acc_rank/len(ranks), 2)

def get_inferred_rank(prediction, label):
    rank = np.where(prediction==label)[0]
    return len(prediction)-rank