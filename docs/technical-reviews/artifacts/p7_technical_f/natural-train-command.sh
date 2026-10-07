mkdir -p outputs/natural-my-v4
cat > outputs/natural-my-v4/train.sh <<'BASH'
.venv-natural/bin/python scripts/natural_assistant.py train \
  --output outputs/natural-my-v4/train \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --steps 2077 --learning-rate 0.00003 \
  --lora-rank 8 --gradient-accumulation 2 \
  --checkpoint-every 25 --checkpoint-steps 1039,2077 \
  --max-seconds 3300
BASH
bash outputs/natural-my-v4/train.sh
