import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os


from src.model import SketchCNN

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,)),
])

def load_classes():
    return 

def predict():
    return

if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))