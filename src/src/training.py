import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

#custom cnn model
from src.model import SketchCNN

transform = transforms.Compere([
    transforms.Grayscle(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,)),
])

dataset = datasets.ImageFolder("data//quickdraw_64//", transform = transform)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size= 128, shuffle = True)
test_loader = DataLoader(test_ds, batch_size= 128)

#hard coded class size to dataset
model = SketchCNN(num_classes=345)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)