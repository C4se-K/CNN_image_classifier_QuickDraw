model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        output = model(images)
        preds = torch.argmax(output, dim=1)
        print(preds[:10])
        break
