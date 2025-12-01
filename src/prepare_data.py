import os
import numpy as np
from torch.utils.data import Dataset
import torch

class QuickDrawNPY(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ])

        self.classes = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.data = []
        self.labels = []

        for class_idx, path in enumerate(self.files):
            arr = np.load(path)
            self.data.append(arr)
            self.labels += [class_idx] * len(arr)

        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(28, 28).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
