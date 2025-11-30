import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os

#custom cnn model
from src.model import SketchCNN

def main():
    transform = transforms.Compose([
        transforms.Grayscale(),
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

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch+1} --- loss: {loss.item():.4f}")

    save_path = os.path.join(OUTPUT_DIR, "model_weights.pth")
    torch.save(model.state_dict(), save_path)
    print("model saved")


OUTPUT_DIR = "model//"

if __name__ == "__main__":
    main()