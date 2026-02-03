import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models

"""

██╗░░░██╗██╗████████╗  ██████╗░███████╗░█████╗░░█████╗░██╗░░░░░██╗░░░░░
██║░░░██║██║╚══██╔══╝  ██╔══██╗██╔════╝██╔══██╗██╔══██╗██║░░░░░██║░░░░░
╚██╗░██╔╝██║░░░██║░░░  ██████╔╝█████╗░░██║░░╚═╝███████║██║░░░░░██║░░░░░
░╚████╔╝░██║░░░██║░░░  ██╔══██╗██╔══╝░░██║░░██╗██╔══██║██║░░░░░██║░░░░░
░░╚██╔╝░░██║░░░██║░░░  ██║░░██║███████╗╚█████╔╝██║░░██║███████╗███████╗
░░░╚═╝░░░╚═╝░░░╚═╝░░░  ╚═╝░░░╚═╝╚══════╝░╚════╝░╚═╝░░╚═╝╚══════╝╚══════╝

// Ian Bezerra - 2026 //

--------------------------------

-> Avaliacao de Recall@k usando ViT (ImageNet) para retrieval de imagens

--------------------------------
"""


############// Configuracao //#####################################


imgs_path = '../imgs'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminho para modelo fine-tuned (None para usar pre-treinado)
FINETUNED_PATH = None  # ou "vit_finetuned.pth"


############// ViT Feature Extractor //#####################################


class ViTEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)


############// Coletando imagens e labels //#####################################


print("Coletando imagens...")
image_paths = []
labels = []
class_names = []

for class_folder in sorted(os.listdir(imgs_path)):
    class_path = os.path.join(imgs_path, class_folder)
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
print(f"Total de imagens: {len(image_paths)}")
print(f"Classes: {class_names}")
print(f"Imagens por classe: {[np.sum(labels == i) for i in range(len(class_names))]}")


############// Carregando ViT //#####################################


print("\nCarregando ViT-B/16...")
torch.cuda.empty_cache()
model = ViTEmbedder()

if FINETUNED_PATH and os.path.exists(FINETUNED_PATH):
    print(f"Carregando pesos fine-tuned de: {FINETUNED_PATH}")
    model.load_state_dict(torch.load(FINETUNED_PATH))
else:
    print("Usando pesos pre-treinados (ImageNet)")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


############// Extraindo features //#####################################


print("\nExtraindo features...")
batch_size = 32
all_features = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]

    input_tensors = []
    for img_path in batch_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        input_tensors.append(img_tensor)

    input_batch = torch.stack(input_tensors).to(device)

    with torch.no_grad():
        features = model(input_batch)

    features = features.cpu().numpy()
    all_features.append(features)

    del input_batch, features
    torch.cuda.empty_cache()

    print(f"  Processado: {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

features = np.vstack(all_features)
print(f"Features shape: {features.shape}")


############// Funcoes de Retrieval //#####################################


def cosine_similarity_matrix(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / norms
    return np.dot(normalized, normalized.T)


def compute_recall_at_k(sim_matrix, labels, k):
    n = len(labels)
    hits = 0

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top_k_idx = np.argsort(sims)[::-1][:k]

        query_label = labels[i]
        retrieved_labels = labels[top_k_idx]

        if query_label in retrieved_labels:
            hits += 1

    return hits / n


############// Avaliacao //#####################################


print("\nCalculando matriz de similaridade...")
sim_matrix = cosine_similarity_matrix(features)

print("\n" + "="*60)
print("  RESULTADOS - RECALL@K (ViT)")
print("="*60)

for k in [1, 2, 5]:
    recall = compute_recall_at_k(sim_matrix, labels, k)
    print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

print("="*60)


############// Detalhamento por classe //#####################################


print("\n" + "="*60)
print("  RECALL@1 POR CLASSE")
print("="*60)

for class_idx, class_name in enumerate(class_names):
    class_mask = labels == class_idx
    class_indices = np.where(class_mask)[0]

    hits = 0
    for i in class_indices:
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top_idx = np.argmax(sims)

        if labels[top_idx] == class_idx:
            hits += 1

    recall = hits / len(class_indices)
    print(f"  {class_name}: {recall:.4f} ({recall*100:.2f}%)")

print("="*60)
