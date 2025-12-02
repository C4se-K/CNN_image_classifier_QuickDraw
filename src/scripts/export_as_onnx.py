import torch
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SRC_PATH = os.path.join(ROOT, "src")
sys.path.append(SRC_PATH)

from model import SketchCNN

model = SketchCNN()

pth_path = os.path.join(ROOT, "models", "10k_20e_basic_73.pth")
state = torch.load(pth_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 1, 28, 28)

onnx_out = os.path.join("export/", "model.onnx")

torch.onnx.export(
    model,
    dummy,
    onnx_out,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

