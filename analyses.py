# analysis.py
import numpy as np
from gcn_base import GCNClassifier
from main import folds 
import matplotlib.pyplot as plt


def experiment_jaccard_thresholds(classifier, thresholds, test_index, train_index, features, labels):
    accurracies = {}
    for limiar in thresholds:
        print(f'\nExperimentando Jaccard com limiar: {limiar}')
        classifier.prepare(test_index, train_index, features, labels, analy=True)
        classifier.create_graph('jaccard', limiar=limiar)
        embeddings, pred = classifier.train_and_predict()
        test_labels = [labels[i] for i in test_index]
        acc = sum(1 for i, p in enumerate(pred) if test_labels[i] == p) / len(pred)
        accurracies[limiar] = acc
        print(f'Acurácia com limiar {limiar}: {acc*100:.2f}%\n')
    return accurracies

def experiment_rbo_thresholds(classifier, thresholds, test_index, train_index, features, labels):
    accuracies= {}
    for limiar in thresholds:
        print(f'Experimentando RBO com limiar: {limiar}')
        classifier.prepare(test_index, train_index, features, labels, analy=True)
        classifier.create_graph('RBO')

if __name__ == "__main__":
   
    features = np.load('features.npy')
    with open('list.txt', 'r') as file:
        dataset_elements = [line.strip() for line in file.readlines()]
    class_size = 80
    labels = [i // class_size for i in range(len(dataset_elements))]
    
    from main import run_ball_tree
    rks = run_ball_tree(features, k=100)
    
    from main import folds
    test_index, train_index = folds[0]

    ks= [5,10,20,40]
    results=[]
    for l in ks:
        print(f'Utilizando K= {l}')
        clf = GCNClassifier('gcn-net', rks, pN=len(labels), number_neighbors=l)
        thresholds = [0.2, 0.4, 0.6, 0.8]
        results.append(experiment_jaccard_thresholds(clf, thresholds, test_index, train_index, features, labels))
    print("Resultados finais:", results)


    Z= np.zeros((len(ks), len(thresholds)))
    for i, res in enumerate(results):
        for j, t in enumerate(thresholds):
            Z[i,j]= res[t]  
    K, T= np.meshgrid(ks, thresholds, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(K, T, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Número de k")
    ax.set_ylabel("Limiar")
    ax.set_zlabel("Acurácia")
    ax.set_title("Superfície 3D: k, Acurácia e Limiar")
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    plt.show()