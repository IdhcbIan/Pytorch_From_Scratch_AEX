import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models

class ViTEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.vit(x)
