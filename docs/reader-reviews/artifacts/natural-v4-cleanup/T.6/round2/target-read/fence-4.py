import json
from tiny_perceptron.data import load_jsonl
from tiny_perceptron.training import load_checkpoint
from scripts.evaluate import evaluate

records = load_jsonl("data/generated/attributes-sft/validation.jsonl")
before, _ = load_checkpoint("checkpoints/attributes.pt", "cpu")
after, _ = load_checkpoint("checkpoints/vision.pt", "cpu")
for label, language in [("before", before), ("after", after.language)]:
    report = evaluate(language, records, mode="sft", max_new_tokens=24)
    print(label, json.dumps(report, ensure_ascii=False))
