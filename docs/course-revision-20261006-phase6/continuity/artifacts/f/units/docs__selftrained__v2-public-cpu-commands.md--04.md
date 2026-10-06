## 圖片：三類服飾辨識

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/vision_clothing.messages.json \
  --task vision_clothing \
  --device cpu \
  --max-new-tokens 128 \
  --image images/vision/c98cb897d639c880fde261eafc6d953d.png \
  --image-layout docs/selftrained/examples/v2/vision_clothing.layout.json \
  --threads 2
```

