import os
import sys
import time
import torch
import natsort
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""

██████╗░██╗███╗░░██╗░█████╗░██╗░░░██╗██████╗░
██╔══██╗██║████╗░██║██╔══██╗██║░░░██║╚════██╗
██║░░██║██║██╔██╗██║██║░░██║╚██╗░██╔╝░░███╔═╝
██║░░██║██║██║╚████║██║░░██║░╚████╔╝░██╔══╝░░
██████╔╝██║██║░╚███║╚█████╔╝░░╚██╔╝░░███████╗
╚═════╝░╚═╝╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚══════╝

// Ian Bezerra - 2026 //


--------------------------------

-> Com BatchSize grande + PreCarregar Imagens em RAM de CPU + Paralel workers para "criar" o nosso Batch, de RAM para VRAM. 

--------------------------------
-> Este codigo carregamos um ViT(Visual tranformer) chamado DinoV2 que foi pre-treinado pela META
    e extraimos tokens(CLS) para diversas imagens!! Assim, podemos comparar e operar em cima dessas representacoes.
    Neste caso famos fazer um exemplo interativo de Retrieval, onde para cada imagem:

--------------------------------

    Imagem_Query       -> Modelo    -> Representacao_Query
                                                  |----------------------> Comparamos e trazemos as k mais similares
    Galeria_De_Imagens -> Modelo    -> Representacoes_Imagens_Galeria

--------------------------------
"""


############// Selecionando as imagens!! //#####################################


# Diretório com as imagens
#os.chdir(image_dir)
imgs_path = './imgs'  # Caminho onde as imagens estão armazenadas
images = natsort.natsorted(os.listdir(imgs_path))  # Lista ordenada de imagens

# Inicialização da lista de features e arquivo de saída
#features = []  # Lista para armazenar as features extraídas
dataset_elements = []  # Lista auxiliar para os nomes das imagens



############// Preprocessando!! //#####################################


# Paths das imagens
image_paths = []

ini_p = time.time()
for i, img in enumerate(images):
    if ".jpg" not in img:  # Ignora arquivos que não sejam imagens .jpg
        continue

    # Salva o nome da imagem no arquivo de texto
    dataset_elements.append(img)

    # Define o caminho completo da imagem e adiciona à lista
    img_path = os.path.join(imgs_path, img)
    image_paths.append(img_path)

    # Log a cada 250 imagens processadas
    if i % 250 == 0:
        print(f"{i} images collected!")

end_p = time.time()



############// Inferencia do modelo //#####################################



# Clear CUDA cache before starting
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14")
dinov2_vits14.to(device)
dinov2_vits14.eval()

# Tranfomrando imagens como no arquiovo original
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, preprocessed_images):
        self.preprocessed_images = preprocessed_images

    def __len__(self):
        return len(self.preprocessed_images)

    def __getitem__(self, idx):
        return self.preprocessed_images[idx]


preprocessed_images = []
print("Started preprocessing")
start_preprocessing = time.perf_counter()
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    preprocessed_images.append(img_tensor)
end_preprocessing = time.perf_counter()
print("Ended preprocessing")


# Start the timer
start = time.perf_counter()


DataLoading_Times = []
Inference_Times = []


# Process images in smaller batches
batch_size = 256  # Adjust based on your GPU memory
num_workers = 4
all_features = []

image_dataset = ImageDataset(preprocessed_images)
image_loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=num_workers,
                          persistent_workers=True, pin_memory=True)
image_loader_iter = iter(image_loader)

for i in range(len(image_loader)):
   
    #Data Loading Start
    start_dataLoading = time.perf_counter()
    
    # Carregando o batch atual com DataLoader
    input_batch = next(image_loader_iter)
    input_batch = input_batch.to(device, non_blocking=True)

    #Data Loading End
    end_dataLoading = time.perf_counter()
    DataLoading_Times.append(end_dataLoading - start_dataLoading)

    #Inference Time Start
    torch.cuda.synchronize()
    start_inference = time.perf_counter()

    with torch.no_grad():
        features = dinov2_vits14(input_batch)


    #Inference Time Start
    torch.cuda.synchronize()
    end_inference = time.perf_counter()
    Inference_Times.append(end_inference - start_inference)
        
    # Move results to CPU immediately to free GPU memory
    features = features.cpu().numpy()
    all_features.append(features)
    
    # Clear some memory
    del input_batch, features
    torch.cuda.empty_cache()

features_batch = np.vstack(all_features)


# End the timer
end = time.perf_counter()


############// Imprimindo resultados //#####################################
t1 = sum(DataLoading_Times)
t2 = sum(Inference_Times) / len(Inference_Times)

print(f"Total dataloading time: {t1} seconds")
print(f"Average inference time: {t2} seconds")
print(f"Preprocessing time: {end_preprocessing - start_preprocessing} seconds")
print(f"Total time: {end - start} seconds")
