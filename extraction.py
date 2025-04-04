# extraction.py
import os
import tarfile
import requests
import numpy as np
import natsort
import torch
from PIL import Image
import pretrainedmodels
import pretrainedmodels.utils as utils

# URL e caminhos
url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
project_dir = os.getcwd()
download_dir = os.path.join(project_dir, 'content')
download_path = os.path.join(download_dir, '17flowers.tgz')
extract_path = os.path.join(download_dir, 'extracted')

os.makedirs(extract_path, exist_ok=True)

def download_dataset(url, download_path):
    try:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            print(f'Diretório foi criado {download_dir}')
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-lenght', 0))
        block_size = 1024
        print('Downloanding...')
        with open(download_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
        print('Download Completed!')
    except Exception as e:
        print(f'An error occured during download: {e}')

def extract_tgz(tar_path, extract_to):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
            print(f"Arquivo extraído para: {extract_to}")
    except Exception as e:
        print(f'Erro ao extrair {e}')

def extract_features(path_img, model):
    tf_img = utils.TransformImage(model)
    load_img = utils.LoadImage()
    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    input_var = torch.autograd.Variable(input_tensor, requires_grad=False)
    features = model(input_var)
    features = features.data.cpu().numpy().tolist()[0]
    return features

def processamento(model, imgs_path):
    # Lista de imagens e extração das features
    images = natsort.natsorted(os.listdir(imgs_path))
    features = []
    dataset_elements = []
    with open('list.txt', 'w') as f:
        for i, img in enumerate(images):
            if '.jpg' not in img:
                continue
            print(img, file=f)
            dataset_elements.append(img)
            img_full_path = os.path.join(imgs_path, img)
            if i % 250 == 0:
                print(f'{i} images processed!')
            features.append(extract_features(img_full_path, model))
    features = np.array(features)
    np.save("features.npy", features)
    print('Done!')
    return features, dataset_elements

if __name__ == "__main__":
    # Exemplo de uso: descomente as linhas abaixo para realizar o download e extração.
    # download_dataset(url, download_path)
    # extract_tgz(download_path, extract_path)

    # Inicializa o modelo
    model_name = 'alexnet'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()
    model.last_linear = pretrainedmodels.utils.Identity()

    imgs_path = os.path.join(download_dir, 'extracted', 'jpg')
    # Chama o processamento para extrair features
    processamento(model, imgs_path)
