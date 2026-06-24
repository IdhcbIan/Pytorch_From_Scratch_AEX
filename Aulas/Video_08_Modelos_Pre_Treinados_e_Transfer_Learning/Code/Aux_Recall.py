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
 AUX Functions to make the Cos Sim and Recall@k
"""



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
