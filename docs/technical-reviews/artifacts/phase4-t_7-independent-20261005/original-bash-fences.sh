.venv/bin/python scripts/prepare_data.py --kind preference
.venv/bin/python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt

.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
.venv/bin/python scripts/infer.py checkpoints/preferred.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0

.venv/bin/python scripts/fetch_training_assets.py --asset ultrafeedback-dpo
.venv/bin/python -m scripts.course_experiments.run --experiment dpo --device cuda

.venv/bin/python -m scripts.course_experiments.run --experiment posttraining --device cpu
