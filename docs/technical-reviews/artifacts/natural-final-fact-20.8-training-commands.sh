.venv-natural/bin/python -c "import torch; print('GPU可用：', torch.cuda.is_available()); print('支援bfloat16：', torch.cuda.is_bf16_supported())"

.venv-natural/bin/python scripts/fetch_natural_data.py --list
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/manifest.json --output data/natural
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/manifest.json --output data/natural --verify

.venv-natural/bin/python scripts/natural_assistant.py prepare \
  --output outputs/natural-my-models \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16

mkdir -p outputs/natural-my-experiment
cat > outputs/natural-my-experiment/train.sh <<'SH'
.venv-natural/bin/python scripts/natural_assistant.py train \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --output outputs/natural-my-experiment/train \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 \
  --steps 180 --learning-rate 1e-4 \
  --lora-rank 8 --gradient-accumulation 2 --seed 42 \
  --checkpoint-every 25 --max-seconds 3300 \
  --local-files-only > outputs/natural-my-experiment/train.stdout.json
SH
bash outputs/natural-my-experiment/train.sh

.venv-natural/bin/python scripts/natural_assistant.py train \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/resumed \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --steps 180 --learning-rate 1e-4 \
  --lora-rank 8 --gradient-accumulation 2 --seed 42 \
  --checkpoint-every 25 --max-seconds 3300 --local-files-only

.venv-natural/bin/python scripts/natural_assistant.py validation \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/validation \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --max-new-tokens 384 --seed 42 --max-seconds 3300 \
  --split validation --local-files-only

sha256sum \
  outputs/natural-my-experiment/train/adapter/adapter_model.safetensors \
  outputs/natural-my-experiment/train/adapter/adapter_config.json \
  docs/natural-assistant/manifest.json \
  outputs/natural-my-experiment/validation/result.json \
  > outputs/natural-my-experiment/chosen-before-test.sha256

.venv-natural/bin/python scripts/natural_assistant.py evaluate \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/final \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --max-new-tokens 384 --seed 42 --max-seconds 3300 \
  --split test --local-files-only
