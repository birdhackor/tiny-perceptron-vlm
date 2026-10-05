from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone

model = CapstoneModel()
with TemporaryDirectory() as directory:
    path = Path(directory) / "example.pt"
    save_capstone(path, model, stage="sft", step=0, inference_only=True)
    loaded, metadata = load_capstone(path)
    print("格式", metadata["format_version"], "訓練步數", metadata["step"])
    print("只供推論", metadata["inference_only"], "參數相同", model.description() == loaded.description())
