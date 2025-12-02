QuickSketch CNN — 28×28 Sketch Classification

This project implements a compact convolutional neural network for
classifying 28×28 grayscale drawings from the Quick, Draw! dataset (or
subsets exported in .npy format). The codebase handles dataset
ingestion, training, evaluation, and exporting the model.

The goal is straightforward: learn a fast, reasonably expressive
architecture that performs well on low-resolution synthetic line
drawings without depending on heavyweight backbones.

1. Project Structure

. │ inference.py │ model.py │ prepare_data.py │ training.py │
├───scripts │ │ check_torch_cuda.py │ │ export_as_onnx.py │ │
prepare_subset.py │ └───pycache └───pycache

-   model.py — Defines the SketchCNN architecture.
-   prepare_data.py — Contains the QuickDrawNPY dataset class.
-   training.py — Training loop, evaluation function, checkpoint saving.
-   inference.py — (Optional) run trained model on a single image or
    batch.
-   scripts/ — Utility tools for subset generation, CUDA diagnostics,
    and ONNX export.

2. Dataset Format

The project expects a directory containing one .npy file per class,
where each file holds a NumPy array of shape:

(num_samples, 784) # flattened 28x28 grayscale images

Example structure:

data/ └── quickdraw_subset_10k/ ├── airplane.npy ├── apple.npy ├──
backpack.npy …

During loading:

-   Arrays are concatenated into a single dataset.
-   Each sample is reshaped to (1, 28, 28) and normalized to [0,1].
-   Optionally, transforms apply after conversion to a tensor.
-   Class labels are derived from the filename order.

3. The CNN Architecture

SketchCNN is a deliberately compact but layered model. It uses three
convolutional blocks, aggressive feature expansion, and lightweight
classification.

Feature Extractor

-   Early layers use 64→128→256 channels.
-   Each block uses Conv → BatchNorm → LeakyReLU repeated twice.
-   MaxPool reduces spatial resolution in the first two blocks; a final
    AvgPool compresses features before flattening.
-   LeakyReLU avoids the dead-ReLU problem common in sparse line
    drawings.

After the final pooling, the feature maps have shape 256×4×4.

Classifier

-   A single hidden layer of size 256, followed by dropout.
-   Final linear layer outputs num_classes.

4. Training Procedure

Training is configured in training.py.

Setup

-   Loss: CrossEntropy
-   Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
-   Scheduler: StepLR (step_size=10, gamma=0.5)
-   Epochs: 20
-   Batch size: 256
-   Device: CUDA if available

Splitting

The dataset is split 90/10 into train/test using random_split.

Transforms include normalization:

Normalize(mean=0.5, std=0.5)

Loop

Each epoch:

1.  Full train pass.
2.  Validation accuracy computed with evaluate().
3.  Scheduler step.

Model is saved to:

models/model_weights.pth

5. Running Training

python training.py

Ensure the dataset path in training.py points to your .npy directory.

6. Inference

Example usage:

model = SketchCNN(num_classes=345)
model.load_state_dict(torch.load(“models/model_weights.pth”))
model.eval()

img: tensor of shape (1, 1, 28, 28)

pred = model(img).argmax(dim=1).item()

7. ONNX Export

python scripts/export_as_onnx.py

8. Subset Preparation

python scripts/prepare_subset.py

9. CUDA Diagnostics

python scripts/check_torch_cuda.py

10. Requirements

-   Python 3.10+
-   PyTorch
-   torchvision
-   NumPy

pip install torch torchvision numpy

11. Notes

-   The dataset loader loads all samples into RAM; large exports may
    require streaming.
-   The model assumes 28×28 inputs.
-   No augmentation is used, though it can be added for robustness.
