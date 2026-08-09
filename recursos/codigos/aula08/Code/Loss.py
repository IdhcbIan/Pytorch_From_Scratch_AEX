import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models



############// Funcao objetivo (Loss) //#####################################



class PositiveAttractionLoss(nn.Module):
    """
    Loss que atrai embeddings da mesma classe e distancia embeddings de classes diferentes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):    # Recebemos input e output e queremos calcular o quao longe estamos...

        # Normalizamos os embeddings para que o produto interno entre dois
        # vetores seja igual a cosine similarity.
        embeddings = nn.functional.normalize(embeddings, dim=1)

        losses = []
        batch_size = embeddings.shape[0]

        for i in range(batch_size):
            for j in range(i + 1, batch_size):   # Passamos sobre i,j em cima de todos os pares.
                
                cosine_similarity = torch.dot(embeddings[i], embeddings[j])   
                if labels[i] == labels[j]:     # Para as imagens que sao da mesma classe dentro do nosso batch
                   
                    # Similaridade cos vai de [-1,1], queremos que ela fique o
                    # mais perto de 1 possivel. Portanto, minimizamos
                    # 1 - sim_cos: quanto mais perto de 1, menor a loss fica.
                    
                    cosine_distance = 1.0 - cosine_similarity   
                    
                    losses.append(cosine_distance)
                else:
                   
                    # Similaridade cos vai de [-1,1], desta vez queremos que ela fique o
                    # mais distante de 1 possivel. Portanto, minimizamos
                    # sim_cos: quanto mais perto de 0, menor a loss fica.
                    
                    # Usamos Relu para ignorar casos onde a similaridade coseno eh menor que 0, relu equivale a max(x, 0)

                    cosine_distance = torch.relu(cosine_similarity)     
                    
                    losses.append(cosine_distance)


        # Se o batch nao tiver nenhum par da mesma classe, nao ha loss.  
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()   # Nossa loss final eh a media da distancia_cosseno dos pares da mesma classe para aquele batch.

