# Retrieving-SQuAD

Ce repo utilise les données de SQuAD pour a partir d'une question et de l'ensemble du corpus retrouver le paragraphe qui répond a la question. Deux modèles ont été implémentés : un avec un encodage par TF-IDF et un deuxième avec un encodage a l'aide de sBERT.

## Installation
Clonez le repo, mettez vous a la base du dossier et installez les librairies necessaires avec cette ligne de commande :  
```pip install -r requirements.txt```

Vous pouvez creer un envirennement virtuel avec virtualenv avant avec les commandes suivantes : 
- Creation de l'environnement : ```python -m venv env_name``` (ou python3 selon ce qui est utilisé)
- Activez l'environnement : ```source env_name/bin/activate```

Desactivez l'environnement après utilisation avec ```deactivate```.

## Scripts
Il y a deux scripts qui sont executables :
- ```evaluate.py``` : renvoie des metriques de réussites sur la totalité du train ou test set
- ```inference.py``` : renvoie la réponse du model pour une question donnée pour le context de train ou test. Pour indiquer la question choisie, le dataset (train ou test) ainsi que l'indice de la question doivent être indiqués dans le fichier de config.

En remplissant un fichier configuration dans le folder configs, on indique quel model on souhaite utiliser ainsi que quel dataset. On peut aussi choisir quelle question pour l'inférence. Le fichier ```configs/config.yml.dist``` est un template pour ces fichiers de configuration.

Pour lancer un de ces deux scripts, on peut lancer dans sa ligne de commande la ligne suivante en remplacant ```script_file``` et ```config_file``` par les noms souhaités :

```python src/script_file.py configs/config_file.yml```

Il y a déjà 4 fichiers de config présents pour appeller les deux modèles implémentés avec les train et test set : 
- ```config_tfidf_train.yml```pour appeller le modèle TF-IDF sur le train set
- ```config_tfidf_test.yml```pour appeller le modèle TF-IDF sur le test set
- ```config_bert_train.yml```pour appeller le modèle BERT sur le train set
- ```config_bert_test.yml```pour appeller le modèle BERT sur le test set


