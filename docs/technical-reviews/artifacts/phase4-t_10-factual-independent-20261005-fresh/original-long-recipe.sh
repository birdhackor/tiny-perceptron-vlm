.venv/bin/python scripts/fetch_training_assets.py --asset gsm8k
.venv/bin/python -m scripts.course_experiments.run --experiment distillation --device cpu
