# analysis.py
import numpy as np
from gcn_base import GCNClassifier
from main import folds  # Supondo que folds seja definido em main.py
# Certifique-se de que features, labels e rks já foram gerados e salvos.

def experiment_jaccard_thresholds(classifier, thresholds, test_index, train_index, features, labels):
    accurracies = {}
    for limiar in thresholds:
        print(f'\nExperimentando com limiar: {limiar}')
        classifier.prepare(test_index, train_index, features, labels, analy=True)
        classifier.create_graph('jaccard', limiar=limiar)
        embeddings, pred = classifier.train_and_predict()
        test_labels = [labels[i] for i in test_index]
        acc = sum(1 for i, p in enumerate(pred) if test_labels[i] == p) / len(pred)
        accurracies[limiar] = acc
        print(f'Acurácia com limiar {limiar}: {acc*100:.2f}%')
    return accurracies

if __name__ == "__main__":
    # Carrega os dados necessários
    features = np.load('features.npy')
    with open('list.txt', 'r') as file:
        dataset_elements = [line.strip() for line in file.readlines()]
    class_size = 80
    labels = [i // class_size for i in range(len(dataset_elements))]
    # Suponha que rks foi gerado (pode ser importado de main.py ou recalculado)
    from main import run_ball_tree
    rks = run_ball_tree(features, k=100)
    
    # Seleciona o primeiro fold para experimentação
    from main import folds
    test_index, train_index = folds[0]
    
    # Inicializa o classificador (lembre-se de ajustar o total de samples)
    clf = GCNClassifier('gcn-net', rks, pN=len(labels), number_neighbors=40)
    thresholds = [0.2, 0.4, 0.6, 0.8]
    results = experiment_jaccard_thresholds(clf, thresholds, test_index, train_index, features, labels)
    print("Resultados finais:", results)

