import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class QuickDrawNPY(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            f.replace(".npy", "") for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.data = []
        self.lengths = []

        for cls in self.classes:
            path = os.path.join(root_dir, cls + ".npy")
            arr = np.load(path, mmap_mode="r")
            self.data.append(arr)
            self.lengths.append(len(arr))

        self.cumsum = np.cumsum(self.lengths)
        self.total = self.cumsum[-1]

        self.to_pil = T.ToPILImage()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        class_idx = np.searchsorted(self.cumsum, idx, side="right")

        if class_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumsum[class_idx - 1]

        arr = self.data[class_idx][local_idx].reshape(28, 28)

        img = self.to_pil(arr)

        if self.transform:
            img = self.transform(img)

        label = class_idx
        return img, label
