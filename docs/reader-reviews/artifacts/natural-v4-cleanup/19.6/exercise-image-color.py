import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, modality_tensors, prepare_batch

splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "image_color")
print("問題", row["user"], "示範回答", row["answer"])
image, audio = modality_tensors(row)
print("實際圖片形狀", tuple(image.shape), "聲學摘要形狀", None if audio is None else tuple(audio.shape))
model = CapstoneModel()
batch, labels = prepare_batch([row])
with torch.no_grad():
    result = model(**batch)
print("回答分數形狀", tuple(result["logits"].shape), "與目標位置對齊", result["logits"].shape[:2] == labels.shape)
print("audio is None", audio is None)
print("audio_features is None", batch["audio_features"] is None)
