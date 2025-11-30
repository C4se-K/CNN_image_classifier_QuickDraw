import os
import numpy as np
from PIL import Image

INPUT_DIR = "data/quickdraw/numpy_bitmap/"
OUTPUT_DIR = "data/quickdraw_64/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".npy"):
        continue

    class_name = file.replace(".npy", "")
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    data = np.load(os.path.join(INPUT_DIR, file))
    data = data.reshape(-1, 28, 28)

    for i, arr in enumerate(data):
        img = Image.fromarray(arr)
        img_resized = img.resize((64, 64), Image.NEAREST)
        img_resized.save(os.path.join(class_dir, f"{class_name}_{i}.png"))

    print("Processed:", class_name)
