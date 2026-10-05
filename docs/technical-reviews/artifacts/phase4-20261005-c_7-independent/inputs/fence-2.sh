git lfs install --local
git lfs pull --include='assets/training/gsm8k-v1.tar.gz' --exclude=''
python scripts/course_experiments/run.py --experiment reasoning --device cpu
