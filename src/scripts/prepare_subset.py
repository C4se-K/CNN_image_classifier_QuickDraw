import os
import numpy as np

INPUT_DIR = "data/quickdraw/numpy_bitmap/"
OUTPUT_DIR = "data/quickdraw_subset/"
N = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".npy"):
        continue

    class_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    print("processing:", fname)

    arr = np.load(class_path)
    subset = arr[:N]
    np.save(out_path, subset)

print("complete")