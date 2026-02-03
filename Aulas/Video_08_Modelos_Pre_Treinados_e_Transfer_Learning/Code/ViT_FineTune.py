import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

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


imgs_path = '../imgs'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametros
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TEMPERATURE = 0.5


############// Dataset //#####################################


class BirdDataset(Dataset):
    def __init__(self, imgs_path, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = []

        for class_folder in sorted(os.listdir(imgs_path)):
            class_path = os.path.join(imgs_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            self.class_names.append(class_folder)
            class_idx = len(self.class_names) - 1

            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith('.jpg'):
                    continue

                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


############// ViT Feature Extractor //#####################################


class ViTEmbedder(nn.Module):
    """ViT que retorna embeddings ao inves de classificacao"""
    def __init__(self):
        super().__init__()
        # Carrega ViT-B/16 pre-treinado no ImageNet
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove a cabeca de classificacao
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)


############// Loss de Atracao Positiva //#####################################


class PositiveAttractionLoss(nn.Module):
    """
    Loss que atrai embeddings de imagens da mesma classe.
    Usa cosine similarity com margem suave.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Normaliza embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)

        # Matriz de similaridade
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Mascara de pares positivos (mesma classe, excluindo diagonal)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        positive_mask.fill_diagonal_(0)

        # Conta pares positivos por amostra
        num_positives = positive_mask.sum(dim=1)

        # Se nao ha pares positivos, retorna 0
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Mascara para todas as amostras exceto si mesmo
        all_mask = torch.ones_like(sim_matrix)
        all_mask.fill_diagonal_(0)

        # Log-sum-exp para estabilidade numerica
        max_sim = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix - max_sim) * all_mask

        # Denominador: soma de todas as similaridades
        log_sum_exp = max_sim.squeeze() + torch.log(exp_sim.sum(dim=1) + 1e-8)

        # Numerador: media das similaridades positivas
        positive_sim_sum = (sim_matrix * positive_mask).sum(dim=1)
        mean_positive_sim = positive_sim_sum / (num_positives + 1e-8)

        # Loss: -log(exp(sim_pos) / sum(exp(sim_all)))
        loss = log_sum_exp - mean_positive_sim

        # Apenas amostras com pares positivos
        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss[valid_mask].mean()


############// Carregando ViT //#####################################


print("Carregando ViT-B/16 (ImageNet)...")
torch.cuda.empty_cache()
model = ViTEmbedder()
model.to(device)
model.train()

print(f"Embedding dimension: 768")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


############// Preparando dados //#####################################


print("\nCarregando dataset...")
dataset = BirdDataset(imgs_path, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print(f"Total de imagens: {len(dataset)}")
print(f"Classes: {dataset.class_names}")


############// Training //#####################################


criterion = PositiveAttractionLoss(temperature=TEMPERATURE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n" + "="*60)
print("  INICIANDO FINE-TUNING")
print("="*60)
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Temperature: {TEMPERATURE}")
print("="*60 + "\n")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        embeddings = model(images)

        # Loss
        loss = criterion(embeddings, labels)

        # Backward
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss.item() if not torch.isnan(loss) else 0
        num_batches += 1

        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    scheduler.step()
    avg_loss = epoch_loss / num_batches
    print(f"\n>>> Epoch {epoch+1}/{EPOCHS} finalizada | Loss medio: {avg_loss:.4f}\n")


############// Salvando modelo //#####################################


save_path = "vit_finetuned.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModelo salvo em: {save_path}")


############// Avaliacao Final //#####################################


print("\n" + "="*60)
print("  AVALIANDO MODELO FINE-TUNED")
print("="*60)

# Transform sem augmentation para avaliacao
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Coleta imagens e labels
print("\nExtraindo features para avaliacao...")
model.eval()

image_paths = []
eval_labels = []

for class_idx, class_name in enumerate(dataset.class_names):
    class_path = os.path.join(imgs_path, class_name)
    for img_name in os.listdir(class_path):
        if img_name.lower().endswith('.jpg'):
            image_paths.append(os.path.join(class_path, img_name))
            eval_labels.append(class_idx)

eval_labels = np.array(eval_labels)

# Extrai features
all_features = []
batch_size = 32

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    input_tensors = []

    for img_path in batch_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = eval_transform(img)
        input_tensors.append(img_tensor)

    input_batch = torch.stack(input_tensors).to(device)

    with torch.no_grad():
        features = model(input_batch)

    all_features.append(features.cpu().numpy())
    del input_batch, features
    torch.cuda.empty_cache()

features = np.vstack(all_features)


# Funcoes de avaliacao
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
        if labels[i] in labels[top_k_idx]:
            hits += 1
    return hits / n


# Calcula metricas
sim_matrix = cosine_similarity_matrix(features)

print("\n" + "="*60)
print("  RESULTADOS - RECALL@K (FINE-TUNED)")
print("="*60)

for k in [1, 2, 5]:
    recall = compute_recall_at_k(sim_matrix, eval_labels, k)
    print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

print("="*60)

# Recall por classe
print("\n  RECALL@1 POR CLASSE:")
for class_idx, class_name in enumerate(dataset.class_names):
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
