import os
import sys
import time
import torch
import natsort
import numpy as np
from PIL import Image
from torchvision import transforms
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

# Process images in smaller batches
batch_size = len(image_paths) // 10  # Adjust based on your GPU memory
all_features = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    
    # Processando imagens do batch atual
    input_tensors = []
    for img_path in batch_paths:
        # Carregando e transformando cada imagem
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        input_tensors.append(img_tensor)

    # Empilhando os tensores do batch atual
    input_batch = torch.stack(input_tensors)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        features = dinov2_vits14(input_batch)
        
    # Move results to CPU immediately to free GPU memory
    features = features.cpu().numpy()
    all_features.append(features)
    
    # Clear some memory
    del input_batch, features
    torch.cuda.empty_cache()

features_batch = np.vstack(all_features)



############// Funcoes de Retrieval //#####################################


# Similaridade Coseno!! A metrica de similaridade que vamos usar... 
#criamos uma matriz nxn

def cosine_similarity(query_feat, gallery_feats):
    # Normalizando os vetores
    query_norm = query_feat / np.linalg.norm(query_feat)
    gallery_norms = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    # Similaridade de cosseno = produto escalar dos vetores normalizados
    similarities = np.dot(gallery_norms, query_norm)
    return similarities


def retrieve_similar_images(query_idx, features, k=5):
    query_feat = features[query_idx]
    similarities = cosine_similarity(query_feat, features)

    # Ordenar por similaridade (maior primeiro) e pegar os top-k
    # Ignoramos o primeiro resultado pois e a propria imagem query
    top_k_indices = np.argsort(similarities)[::-1][:k+1]
    top_k_similarities = similarities[top_k_indices]

    return top_k_indices, top_k_similarities


def display_retrieval_results(query_idx, top_k_indices, top_k_similarities, image_paths, k):
    # Calculando o layout do grid
    n_images = k + 1  # Query + k resultados
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Garante que axes seja sempre 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plotando a imagem query
    query_img = Image.open(image_paths[query_idx])
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f"QUERY (idx: {query_idx})", fontsize=12, fontweight='bold', color='blue')
    axes[0, 0].axis('off')

    # Plotando as imagens similares
    for i, (idx, sim) in enumerate(zip(top_k_indices[1:], top_k_similarities[1:]), start=1):
        row = i // n_cols
        col = i % n_cols

        img = Image.open(image_paths[idx])
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"#{i} | idx: {idx}\nSim: {sim:.4f}", fontsize=10)
        axes[row, col].axis('off')

    # Esconde eixos vazios
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.suptitle("DINOv2 Image Retrieval", fontsize=14, fontweight='bold', y=1.02)
    plt.show()


############// Loop Interativo de Retrieval //#####################################

#Quantas imagens trazemos!!
k = 7

print("\n" + "="*60)
print("  DINOv2 RETRIEVAL INTERATIVO")
print("="*60)
print(f"  Total de imagens na galeria: {len(image_paths)}")
print(f"  Dimensao das features: {features_batch.shape[1]}")
print("="*60)

while True:
    print("\n[Digite 'q' para sair]")

    # Pergunta o indice da imagem query
    query_input = input("Digite o numero da imagem query (0 a {}): ".format(len(image_paths) - 1))

    if query_input.lower() == 'q':
        print("Encerrando...")
        break

    try:
        query_idx = int(query_input)

        if query_idx < 0 or query_idx >= len(image_paths):
            print(f"Erro: Indice deve estar entre 0 e {len(image_paths) - 1}")
            continue

    except ValueError:
        print("Erro: Digite um numero valido!")
        continue


    print(f"\nBuscando {k} imagens mais similares a imagem {query_idx}...")

    # Realiza o retrieval
    top_k_indices, top_k_similarities = retrieve_similar_images(query_idx, features_batch, k)

    print(f"Query: {dataset_elements[query_idx]}")
    print("\nResultados:")
    for i, (idx, sim) in enumerate(zip(top_k_indices, top_k_similarities)):
        marker = " <- QUERY" if i == 0 else ""
        print(f"  {i}. [{idx}] {dataset_elements[idx]} (sim: {sim:.4f}){marker}")

    # Exibe o grid de imagens
    display_retrieval_results(query_idx, top_k_indices, top_k_similarities, image_paths, k)


###############################################################################
