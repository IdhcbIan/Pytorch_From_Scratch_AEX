import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models


from ViT_Embedder import ViTEmbedder


"""

██╗░░░██╗██╗████████╗  ███████╗██╗███╗░░██╗███████╗████████╗██╗░░░██╗███╗░░██╗███████╗
██║░░░██║██║╚══██╔══╝  ██╔════╝██║████╗░██║██╔════╝╚══██╔══╝██║░░░██║████╗░██║██╔════╝
╚██╗░██╔╝██║░░░██║░░░  █████╗░░██║██╔██╗██║█████╗░░░░░██║░░░██║░░░██║██╔██╗██║█████╗░░
░╚████╔╝░██║░░░██║░░░  ██╔══╝░░██║██║╚████║██╔══╝░░░░░██║░░░██║░░░██║██║╚████║██╔══╝░░
░░╚██╔╝░░██║░░░██║░░░  ██║░░░░░██║██║░╚███║███████╗░░░██║░░░╚██████╔╝██║░╚███║███████╗
░░░╚═╝░░░╚═╝░░░╚═╝░░░  ╚═╝░░░░░╚═╝╚═╝░░╚══╝╚══════╝░░░╚═╝░░░░╚═════╝░╚═╝░░╚══╝╚══════╝

// Ian Bezerra - 2026 //

--------------------------------

-> Fine-tuning do ViT (ImageNet) com loss de atracao positiva
   Imagens da mesma classe sao atraidas no espaco de embeddings

--------------------------------
"""


############// Configuracao //#####################################


script_dir = os.path.dirname(os.path.abspath(__file__))
imgs_path = os.path.join(script_dir, '..', 'imgs')
train_path = os.path.join(imgs_path, 'Train')
test_path = os.path.join(imgs_path, 'Test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Hiperparametros

EPOCHS = 10    # Numero de epocas do Treinamento

BATCH_SIZE = 16     # Tamanho do nosso batch de imagens por foward-backward

LEARNING_RATE = 1e-4     # Passo do Optimizer


############// Dataset //#####################################


def collect_images(split_path):
    image_paths = []
    labels = []
    class_names = []

    for class_folder in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_names.append(class_folder)
        class_idx = len(class_names) - 1

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith('.jpg'):
                continue

            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(class_idx)

    labels = np.array(labels)
    return image_paths, labels, class_names


print("\nCarregando dataset de treino...")
image_paths, labels, class_names = collect_images(train_path)

print(f"Total de imagens: {len(image_paths)}")
print(f"Classes: {len(class_names)}")



############// ViT Feature Extractor //#####################################



# Deixamos nossa Loss no outro arquivo, Loss.py

from Loss import PositiveAttractionLoss



############// Carregando ViT //#####################################


print("Carregando ViT-B/16 (ImageNet)...")

torch.cuda.empty_cache()

model = ViTEmbedder()    # Usamos apenas um ViT nao DinoV2 como na aula 7.

model.to(device)
model.train()

print(f"Embedding dimension: 768")


# Nomalizacao padrao ImageNet

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])






############// Definindo parametros de treinamento //#####################################

"""
Fora a arquitetura do modelo, essas sao as coisas mais importantes de um pipeline de treinamento.

Primeiro, nossa funcao objetivo, nossa loss.

Segundo, nosso optimizer, navegamos nossa funcao objetico, 
    buscando os parametros do modelo que minimizam a nossa loss

Terceiro, nosso scheduler, como vamos reduzir nosso tamanho de passo, para que 
      a loss consiga convergir de forma mais suave. 
"""





criterion = PositiveAttractionLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)







############// Training //#####################################


print("\n" + "="*60)
print("  INICIANDO FINE-TUNING")
print("="*60)
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print("="*60 + "\n")


for epoch in range(EPOCHS):
    
    epoch_loss = 0.0
    num_batches = 0
    
    shuffled_indices = np.random.permutation(len(image_paths))
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_indices = shuffled_indices[start:end]

        batch_images = []
        batch_labels = []

        # Fazendo o Preprocessamento do batch. 
        for idx in batch_indices:
            img = Image.open(image_paths[idx]).convert('RGB')
            img = transform(img)
            batch_images.append(img)
            batch_labels.append(labels[idx])

        
        images = torch.stack(batch_images).to(device)     # Temos nossas imagens em tensores.
        batch_labels = torch.tensor(batch_labels, device=device)   

        optimizer.zero_grad()    # Em preparacao para o backwardpass

        # Forward
        embeddings = model(images)    # Foward Pass!!

        # Loss
        loss = criterion(embeddings, batch_labels)    # Cauculamos a Loss!!

        # Backward
        loss.backward()   # Cauculamos a o gradiente pelo grafo computacional, backpropagation.
        optimizer.step()     # Optimizer step

        epoch_loss += loss.item() if not torch.isnan(loss) else 0
        num_batches += 1

        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f}")

        scheduler.step()    # Scheduler step para reduzir o lr do optimizer. (Diminuir o tamanho do passo)
    
    avg_loss = epoch_loss / num_batches
    print(f"\n>>> Epoch {epoch+1}/{EPOCHS} finalizada | Loss medio: {avg_loss:.4f}\n")







############// Salvando modelo //#####################################


save_path = "vit_finetuned.pth"
torch.save(model.state_dict(), save_path)     # Salvamos os pesos do modelo treinado!
print(f"\nModelo salvo em: {save_path}")



############// Avaliacao Final //#####################################


print("\n" + "="*60)
print("  AVALIANDO MODELO FINE-TUNED")
print("="*60)


# Carrega imagens do conjunto de teste
print("\nCarregando dataset de teste...")

model.eval()

test_image_paths, eval_labels, test_class_names = collect_images(test_path)


print(f"Total de imagens de teste: {len(test_image_paths)}")
print(f"Classes de teste: {len(test_class_names)}")


if len(test_image_paths) == 0:
    raise RuntimeError("Nenhuma imagem em imgs/Test. Rode Code/Make_Test_Split.py primeiro.")


print("\nExtraindo features para avaliacao...")



# Extrai features (Do split de teste)

all_features = []
batch_size = 32

for i in range(0, len(test_image_paths), batch_size):
    
    batch_paths = test_image_paths[i:i+batch_size]
    input_tensors = []

    for img_path in batch_paths:    # Carregando o Batch
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        input_tensors.append(img_tensor)

    input_batch = torch.stack(input_tensors).to(device)  # Movendo o batch do test split para GPU


    with torch.no_grad():    # Note que agora usamos no_grad() assim, pois nao vamos backprop nos dados de teste

        features = model(input_batch)    # Inferencia.


    all_features.append(features.cpu().numpy())
    del input_batch, features
    torch.cuda.empty_cache()


# Temos agora uma nossa lista de features igual na ultima aula!! 
#       No entanto, esta inferencia foi feita com a versao treinada do nosso modelo

features = np.vstack(all_features)    # Finalmente, temos as features de teste, igual vimos na aula 7.



from Aux_Recall import compute_recall_at_k
from Aux_Recall import cosine_similarity_matrix




# Calcula metricas
sim_matrix = cosine_similarity_matrix(features)



print("\n" + "="*60)
print("  RESULTADOS - RECALL@K (FINE-TUNED)")
print("="*60)




for k in [1, 2, 3]:
    
    recall = compute_recall_at_k(sim_matrix, eval_labels, k)
    print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

print("="*60)




#-----------------------------------------------------------------------------------------------------


# Recall por classe
print("\n  RECALL@1 POR CLASSE:")


# Classificando por classes

for class_idx, class_name in enumerate(test_class_names):
    
    class_indices = np.where(eval_labels == class_idx)[0]
    hits = 0
    
    for i in class_indices:
        
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        
        if eval_labels[np.argmax(sims)] == class_idx:
            hits += 1
    
    recall = hits / len(class_indices)
    print(f"    {class_name}: {recall:.4f} ({recall*100:.2f}%)")



print("="*60)
print("\n  FINE-TUNING COMPLETO!")
print("="*60)


#-----------------------------------------------------------------------------------------------------

