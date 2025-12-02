# QuickSketch CNN — 28×28 Sketch Classification

This project implements a compact convolutional neural network for classifying **28×28 grayscale drawings** from the **Quick, Draw!** dataset (or subsets exported in `.npy` format). The codebase handles dataset ingestion, training, evaluation, and exporting the model.

The goal is straightforward: learn a fast, reasonably expressive architecture that performs well on low-resolution synthetic line drawings without depending on heavyweight backbones.

---

## Project Structure
```text
.
│ inference.py
│ model.py
│ prepare_data.py
│ training.py
│
├───scripts
│ │ check_torch_cuda.py
│ │ export_as_onnx.py
│ │ prepare_subset.py
│ └───pycache
└───pycache

```

- **model.py** — Defines the `SketchCNN` architecture.  
- **prepare_data.py** — Contains the `QuickDrawNPY` dataset class.  
- **training.py** — Training loop, evaluation function, checkpoint saving.  
- **inference.py** — Run trained model inference.  
- **scripts/** — Subset generation, CUDA diagnostics, and ONNX export.

---

## Dataset Format

The project expects a directory containing **one `.npy` file per class**, where each file holds:


Example:
data/
└── quickdraw_subset_10k/
    ├── airplane.npy
    ├── apple.npy
    ├── backpack.npy
    ...


During loading:

- All `.npy` arrays are concatenated.
- Each sample becomes a **(1, 28, 28)** float tensor in `[0,1]`.
- Class labels are derived from the sorted filenames.
- Optional transforms apply after tensor conversion.

---

## Model Architecture: SketchCNN

`SketchCNN` is a compact but deep convolutional network tuned for low-resolution doodle-like drawings.

### Feature Extractor

- Convolutional blocks expand channels **64 → 128 → 256**.
- Each block uses repeated **Conv → BatchNorm → LeakyReLU** layers.
- **MaxPool** downsampling in the first two stages; final **AvgPool** compresses to 4×4.  
- LeakyReLU mitigates dead features often seen in sparse line drawings.

Output shape before the classifier: **256×4×4**.

### Classifier

- Flatten → Linear(4096 → 256) → LeakyReLU → Dropout → Linear → logits.
- Final output dimension = number of classes (auto-detected).

---

## Training

Training configuration lives in `training.py`.

### Hyperparameters

- **Loss:** CrossEntropy  
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)  
- **Scheduler:** StepLR (step_size=10, gamma=0.5)  
- **Epochs:** 20  
- **Batch size:** 256  
- **Device:** CUDA if available

### Data Split

Dataset split into 90% train / 10% test via `random_split`.

Normalization: ```bash Normalize(mean=0.5, std=0.5) ```

### Loop

Each epoch:

1. Full training pass with gradient updates.
2. Validation accuracy computed via `evaluate()`.
3. Scheduler step.

Training prints epoch loss and validation accuracy.

### Saving

After training, weights are saved to: ```bash models/model_weights.pth ```


---

## Running Training

```bash
python training.py
















    
