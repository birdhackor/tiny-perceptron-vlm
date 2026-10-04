import torch
from tiny_perceptron.capstone import build_dataset, modality_tensors, prepare_batch
from tiny_perceptron.data import ByteTokenizer
splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "image_color")
image, audio = modality_tensors(row)
batch, labels = prepare_batch([row])
tok = ByteTokenizer()
print("素材audio is None", audio is None)
print("batch keys", sorted(batch))
print("圖片標記數", int((batch["ids"] == tok.image_id).sum()))
print("聲音標記數", int((batch["ids"] == tok.audio_id).sum()))
print("聲學欄形狀", tuple(batch["audio_features"].shape))
print("聲學欄全零", bool((batch["audio_features"] == 0).all()))
