# main.py
import os
import numpy as np
import natsort
from sklearn.neighbors import BallTree
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from PIL import Image
import umap.umap_ as umap
from sklearn.manifold import TSNE


features = np.load('features.npy')
with open('list.txt', 'r') as file:
    dataset_elements = [line.strip() for line in file.readlines()]


def run_ball_tree(features, k=100):
    if not isinstance(features, np.ndarray):
        raise ValueError('As features devem estar em numpy')
    if features.ndim != 2:
        raise ValueError('As features devem ser um array 2D no formato (n_samples, n_features)')
    tree = BallTree(features)
    _, rks = tree.query(features, k=k)
    return rks

rks = run_ball_tree(features, k=100)

print(f'Número de imagens: {len(dataset_elements)}')
class_size = 80
labels = [i // class_size for i in range(len(dataset_elements))]


def fold_split(features, labels, n_folds=10):
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    return list(kf.split(features, labels))

folds = fold_split(features, labels, n_folds=10)


def plot_tsne(features, labels=None, perplexity=30, n_components=2, learning_rate=200, n_iter=1000):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10,8))
    if labels is not None:
        scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=0.5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.tight_layout()
    plt.show()

def plot_umap(features, labels=None, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    import umap.umap_ as umap
    if not isinstance(features, np.ndarray):
        raise ValueError("O argumento 'features' deve ser uma matriz NumPy.")
    if labels is not None and len(labels) != features.shape[0]:
        raise ValueError("O tamanho de 'labels' deve corresponder ao número de amostras em 'features'.")
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    umap_results = umap_model.fit_transform(features)
    plt.figure(figsize=(12, 8))
    if n_components == 2:
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Spectral', s=50)
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.figure().add_subplot(projection='3d')
        scatter = ax.scatter(umap_results[:, 0], umap_results[:, 1], umap_results[:, 2], c=labels, cmap='Spectral', s=50)
        ax.set_xlabel("UMAP Dimension 1", fontsize=14)
        ax.set_ylabel("UMAP Dimension 2", fontsize=14)
        ax.set_zlabel("UMAP Dimension 3", fontsize=14)
    else:
        raise ValueError("O valor de 'n_components' deve ser 2 ou 3.")
    plt.title('UMAP Projection', fontsize=18)
    plt.show()


def build_ranked_paths(imgs_dir, dataset_elements, rankings, query, top_n=15):
    return [os.path.join(imgs_dir, dataset_elements[img]) for img in rankings[query][:top_n]]

def display_rk(image_paths_before, image_paths_after):
    num_images = len(image_paths_after)
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(num_images*2, 4))
    if num_images == 1:
        axes = [axes]
    for i, (ax_before, ax_after, img_path_before, img_path_after) in enumerate(zip(axes[0], axes[1], image_paths_before, image_paths_after)):
        image_before = Image.open(img_path_before).resize((100,100), Image.LANCZOS)
        ax_before.imshow(image_before)
        ax_before.axis('off')
        if i == 0:
            ax_before.set_title("Antes")
        image_after = Image.open(img_path_after).resize((100,100), Image.LANCZOS)
        ax_after.imshow(image_after)
        ax_after.axis('off')
        if i == 0:
            ax_after.set_title("Depois")
    plt.tight_layout()
    plt.show()

from gcn_base import GCNClassifier

if __name__ == "__main__":

    test_index, train_index= folds[0]

    clf= GCNClassifier('gcn-net', rks, len(labels), number_neighbors=40)
    clf.prepare(test_index, train_index, features, labels)
    embeddings, pred= clf.train_and_predict()

    embeddings= embeddings.detach().numpy()

    test_labels= [labels[i] for i in test_index]
    acc= sum(1 for i, p in enumerate(pred) if test_labels[i] == p) / len(pred)
    print(f'Acurracy: {acc*100:.2f}%')
    
    plot_tsne(features, labels)
    plot_tsne(embeddings, labels)
    plot_umap(features, labels)
    plot_umap(embeddings, labels)

    imgs_dir = os.path.join(os.getcwd(), 'content', 'extracted', 'jpg')
    query = 1000
    rk_before = build_ranked_paths(imgs_dir, dataset_elements, rks, query)

    new_rks = run_ball_tree(embeddings, k=100)
    rk_after = build_ranked_paths(imgs_dir, dataset_elements, new_rks, query)
    display_rk(rk_before, rk_after)
