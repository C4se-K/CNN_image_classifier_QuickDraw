import os
import numpy as np
from PIL import Image


INPUT_DIR = "data/quickdraw/numpy_bitmap/"
OUTPUT_DIR = "data//quickdraw_64//"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_bin(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    data = data.reshape(-1, 28, 28)
    return data

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".bin"):
        continue

    class_name = file.replace(".bin", "")
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    data = load_bin(os.path.join(INPUT_DIR, file))

    for i, img_array in enumerate(data):
        img = Image.fromarray(img_array)
        img_resized = img.resize((64, 64), Image.NEAREST)

        img_resized.save(os.path.join(class_dir, f"{class_name}_{i}.png"))

    print(f"processed : {class_name}")
