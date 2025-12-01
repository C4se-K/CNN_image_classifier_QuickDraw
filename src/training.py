import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.nn as nn
import os

from model import SketchCNN
from prepare_data import QuickDrawNPY
import scripts.check_torch_cuda

OUTPUT_DIR = "models/"

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    print("starting training...")

    transform = T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])

    dataset = QuickDrawNPY("data/quickdraw_subset_5k/", transform=transform)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    print("preparing dataset splits...")
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=8, pin_memory=True)

    num_classes = len(dataset.classes)
    print("detected classes:", num_classes)

    model = SketchCNN(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    scripts.check_torch_cuda.check()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("training start")

    for epoch in range(40):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_weights.pth"))


if __name__ == "__main__":
    main()
