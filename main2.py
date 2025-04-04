import os
import tarfile
import requests
import numpy as np
import natsort
import torch
from PIL import Image
import pretrainedmodels
import pretrainedmodels.utils as utils
from sklearn.neighbors import BallTree
from sklearn.model_selection import StratifiedKFold
import torch.optim.adam
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

##Importar as imagens

url= 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

project_dir= os.getcwd()
download_dir= os.path.join(project_dir, 'content')
download_path= os.path.join(download_dir, '17flowers.tgz')
extract_path= os.path.join(download_dir, 'extracted')

os.makedirs(extract_path, exist_ok=True)

def download_dataset(url, download_path):
    try:

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            print(f'Diretório foi criado {download_dir}')

        response= requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-lenght', 0))
        block_size= 1024
        print('Downloanding...')
        with open(download_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
        print('Download Completed!')
    except Exception as e:
        print(f'An error occured during download: {e}')

#download_dataset(url, download_path)

def extract_tgz(tar_path, extract_to):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
            print(f"Arquivo extraído para: {extract_to}")
    except Exception as e:
        print(f'Erro ao extrair {e}')

#extract_tgz(download_path, extract_path)



def extract_features(path_img):
    
    tf_img= utils.TransformImage(model)

    load_img= utils.LoadImage()
    input_img= load_img(path_img)

    input_tensor= tf_img(input_img)
    input_tensor= input_tensor.unsqueeze(0)

    input= torch.autograd.Variable(input_tensor, requires_grad=False)

    features= model(input)
    features= features.data.cpu().numpy().tolist()[0]
    return features

model_name= 'alexnet'
model= pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

model.last_linear= pretrainedmodels.utils.Identity()

imgs_path= os.path.join(download_dir, 'extracted', 'jpg')
images= natsort.natsorted(os.listdir(imgs_path))

features= []
print(model_name)
dataset_elements= []

def processamento():
    global features
    global dataset_elements
    f= open('list.txt', 'w+')

    for i, img in enumerate(images):
        if '.jpg' not in img:
            continue

        print(img, file=f)
        dataset_elements.append(img)

        img= os.path.join(imgs_path, img)

        if i % 250 == 0:
            print(f'{i} images processed!')
        
        features.append(extract_features(img))
    
    f.close()

    features= np.array(features)
    np.save("features", features)
    print('Done!')

    return features

#processamento()

#################################################################################################################################

features= np.load('features.npy')

def run_ball_tree(features, k=100):

    if not isinstance(features, np.ndarray):
        raise ValueError('As features devem estar em numpy')
    if features.ndim != 2:
        raise ValueError('As features devem ser um array 2D no formato (n_samples, n_features)')

    tree= BallTree(features)

    _, rks= tree.query(features, k=k)
    
    return rks

rks= run_ball_tree(features, k=100)



with open('list.txt', 'r') as file:
    dataset_elements= file.readlines()

dataset_elements= [element.strip() for element in dataset_elements]
print(len(dataset_elements))
class_size= 80
labels= [i//class_size for i in range(len(dataset_elements))]


def fold_split(features, labels, n_folds=10):

    kf= StratifiedKFold(n_splits=n_folds, shuffle=False)
    res= kf.split(features, labels)
    return list(res)

folds= fold_split(features, labels, n_folds=10)



#########################################################################################################

class Net(torch.nn.Module):
    def __init__(self, pNFeatures, pNNeurons, numberOfClasses):
        super(Net, self).__init__()
        self.conv1= GCNConv(pNFeatures, pNNeurons)
        self.conv2= GCNConv(pNNeurons, numberOfClasses)

    def forward(self, data):
        x, edge_index= data.x, data.edge_index
        x= self.conv1(x, edge_index)
        x= F.relu(x)
        x= F.dropout(x, training=self.training)
        x= self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class GCNClassifier:
    def __init__(self, gcn_type, rks, pN, number_neighbors=40):
        
        self.pK= number_neighbors
        self.pN= pN
        self.rks= rks
        print(self.rks)
        self.pLR= 0.001
        self.pNNeurons= 32
        self.pNEpochs= 50
        self.gcn_type= gcn_type
    
    def prepare(self, test_index, train_index, features, labels):

        print('Creating Masks...')
        self.train_mask= [False] * self.pN
        self.val_mask= [False] * self.pN
        self.test_mask= [False] * self.pN

        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True
        
        self.train_mask= torch.tensor(self.train_mask)
        self.val_mask= torch.tensor(self.val_mask)
        self.test_mask= torch.tensor(self.test_mask)

        print('Set Labels...')
        y= labels
        self.numbersOfClasses= max(y) + 1
        self.y= torch.tensor(y)

        self.x= torch.tensor(features)
        self.pNFeatures= len(features[0])

        self.create_graph('jaccard')


    def compute_jaccard(self, rks, top_k, limiar):
        edge_index= []

        for i in range(len(rks)):
            for j in range(len(rks)):
                b= len(set(rks[i][:top_k]) & set(rks[j][:top_k])) / len(set(rks[i][:top_k]) | set(rks[j][:top_k]))
                if b >= limiar:
                    edge_index.append([i,j])

       
        edge_index= torch.tensor(edge_index)
        self.edge_index= edge_index.t().contiguous()
        


    def create_graph(self, correlation_measure, limiar=0.8):

        if correlation_measure == 'normal':
            print('Making edge list...')

            self.top_k= self.pK
            edge_index=[]

            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2= self.rks[img1][pos]
                    edge_index.append([img1,img2])
            
            edge_index= torch.tensor(edge_index)
            self.edge_index= edge_index.t().contiguous()
    
        if correlation_measure == 'jaccard':
            self.compute_jaccard(self.rks, 100, limiar=limiar)


    def train_and_predict(self):
        print('Loading data object...')

        data= Data(x=self.x.float(), edge_index=self.edge_index, y=self.y, test_mask=self.test_mask, train_mask=self.train_mask, val_mask= self.val_mask)

        model= Net(self.pNFeatures, self.pNNeurons, self.numbersOfClasses)
        optimizer= torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        print('Training')   
        model.train()
        for epoch in range(self.pNEpochs):
            print(f'Training epoch: {epoch}')
            optimizer.zero_grad()
            out= model(data)
            data.y= torch.tensor(data.y, dtype=torch.long)
            loss= F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        print('Evaluating...')
        model.eval()
        _, pred= model(data).max(dim=1)
        pred= torch.masked_select(pred, data.test_mask)

        embeddings= model(data)

        return embeddings, pred.tolist()

clf= GCNClassifier('gcn-net', rks, len(labels), number_neighbors=40)

for test_index, train_index in folds[:1]:
    clf.prepare(test_index, train_index, features, labels)
    embeddings, pred= clf.train_and_predict()


test_labels= [labels[i] for i in test_index]

acc= 0
for i in range(len(pred)):
    if test_labels[i] == pred[i]:
        acc+=1

acc= acc/len(pred)

accuracy_percentage= acc*100

print(f'Acurácia: {accuracy_percentage:.2f}%')




embeddings= embeddings.detach().numpy()

#######################################################################################################################

def plot_tsne(features, labels=None, perplexity=30, n_components=2, learning_rate=200, n_iter=1000):

    tsne= TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity, n_iter=n_iter)
    tsne_results= tsne.fit_transform(features)

    plt.figure(figsize=(10,8))

    if labels is not None:
        scatter= plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Labels')
    
    else:
        plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=0.5)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.tight_layout()
    plt.show()

#plot_tsne(features, labels)
#plot_tsne(embeddings, labels)

############################################################################################################################

def plot_umap(features, labels=None, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    
    if not isinstance(features, np.ndarray):
        raise ValueError("O argumento 'features' deve ser uma matriz NumPy.")
    if labels is not None and len(labels) != features.shape[0]:
        raise ValueError("O tamanho de 'labels' deve corresponder ao número de amostras em 'features'.")
    
    umap_model= umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    umap_results= umap_model.fit_transform(features)

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

#plot_umap(features, labels)
#plot_umap(embeddings, labels)

###################################################################################################################################

new_rks= run_ball_tree(embeddings, k=100)

def build_ranked_paths(imgs_dir, dataset_elements, rankings, query, top_n=15):
    return [imgs_dir + dataset_elements[img] for img in rankings[query][:top_n]]

imgs_dir= 'C:\\Users\\Gabri\\Desktop\\Gabriel\\Programação\\Pyton\\IC-GCN\\content\\extracted\\jpg\\'
query=1000

rk_before= build_ranked_paths(imgs_dir, dataset_elements, rks, query)
rk_after= build_ranked_paths(imgs_dir, dataset_elements, new_rks, query)


def display_rk(image_paths_before, image_paths_after):
    num_images= len(image_paths_after)

    fig, axes= plt.subplots(nrows=2, ncols=num_images, figsize=(num_images*2,4))
    
    if num_images == 1:
        axes= [axes]

    for i, (ax_before, ax_after, img_path_before, img_path_after) in enumerate(zip(axes[0], axes[1], image_paths_before, image_paths_after)):
        image_before= Image.open(img_path_before)
        image_before= image_before.resize((100,100), Image.LANCZOS)
        ax_before.imshow(np.array(image_before))
        ax_before.axis('off')
        
        if i == 0:
            ax_before.set_title("Antes")
        
        image_after= Image.open(img_path_after)
        image_after= image_after.resize((100,100), Image.LANCZOS)
        ax_after.imshow(np.array(image_after))
        ax_after.axis('off')

        if i == 0:
            ax_after.set_title("Depois")
    
    plt.tight_layout()
    plt.show()

display_rk(rk_before, rk_after)


def experiment_jaccard_thresholds(classifier, thresholds, test_index, train_index, features, labels):
    accurracies= []
    for limiar in thresholds:
        print(f'\nExperimentando com limiar: {limiar}')
        classifier.prepare(test_index, train_index, features, labels)
        classifier.create_graph('jaccard', limiar=limiar)
        embeddings, pred= classifier.train_and_predict()

        test_labels= [labels[i] for i in test_index]
        acc= sum(1 for i, p in enumerate(pred) if test_labels[i] == p) / len(pred)
        accurracies[limiar]= acc
        print(f'Acurácia com limiar {limiar}: {acc*100:.2f}%')

    return accurracies

for test_index, train_index in folds[:1]:
    thresholds= [0.2,0.4]
    acc_results= experiment_jaccard_thresholds(clf, thresholds, test_index, train_index, features, labels)
    print("Resultados: ", acc_results)