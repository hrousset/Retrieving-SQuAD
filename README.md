# Retrieving-SQuAD

Ce repo utilise les données de SQuAD pour a partir d'une question et de l'ensemble du corpus retrouver le paragraphe qui répond a la question.

## Installation
pip3 freeze > requirements.txt
```pip install -r requirements.txt```

## Scripts
Il y a deux scripts qui sont executables :
- evaluate : qui renvoie des metriques de réussites sur le train ou test set
- inference : qui renvoie la réponse du model pour une question donnée pour le context de train ou test

En remplissant un fichier config dans le folder configs, on indique quel model on souhaote utiliser ainsi que quel dataset.

Pour lancer un de ces deux scripts, on peut lancer dans sa ligne de commande la ligne suivante en remplacant script_file et config_file par les noms souhaités :
```python src/script_file.py congigs/config_file.yml```


