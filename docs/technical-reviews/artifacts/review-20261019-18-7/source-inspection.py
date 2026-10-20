import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_main = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.in_main = True

    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False

    def handle_data(self, data):
        if self.in_main and data.strip():
            self.parts.append(data.strip())


def fetch(url, phrases):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=False)
    item = {"url": url, "command": command, "returncode": result.returncode,
            "read_at": datetime.now(timezone.utc).isoformat(), "stderr": result.stderr.decode()}
    if result.returncode:
        return item
    raw = result.stdout
    parser = MainText()
    parser.feed(raw.decode())
    text = " ".join(" ".join(parser.parts).split())
    item.update(full_response_sha256=sha(raw), response_bytes=len(raw), excerpts=[])
    for phrase in phrases:
        position = text.find(phrase)
        if position < 0:
            raise ValueError("官方文件找不到指定定位：" + phrase)
        item["excerpts"].append({"locator": phrase, "text": text[max(0, position - 100):position + 950]})
    return item


def functions(raw, names):
    text = raw.decode()
    tree = ast.parse(text)
    lines = text.splitlines()
    selected = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = min([node.lineno] + [n.lineno for n in node.decorator_list])
            selected[node.name] = {"start_line": start, "end_line": node.end_lineno,
                                   "text": "\n".join(lines[start - 1:node.end_lineno])}
    if set(selected) != set(names):
        raise ValueError("來源函式清單不完整")
    return selected


urls = [
    ("https://docs.pytorch.org/docs/2.14/notes/autograd.html",
     ["During the forward pass, an operation", "Evaluation mode is not a mechanism", "Computations in no-grad mode"]),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html",
     ["In this mode, the result", "This API does not apply"]),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html",
     ["Implements stochastic gradient descent", "iterable of parameters", "learning rate"]),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.clone.html",
     ["Returns a copy of"]),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html",
     ["Applies an affine linear transformation", "initialized from"]),
    ("https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html",
     ["def train_knowledge_distillation", "Teacher set to evaluation mode", "we do not change the teacher"]),
]
external = [fetch(url, phrases) for url, phrases in urls]
record_path = ROOT / "docs/course-experiments/results/distillation.json"
record_raw = record_path.read_bytes()
record = json.loads(record_raw)
revision = subprocess.check_output(["git", "rev-parse", "dc8d1ac3"], cwd=ROOT).decode().strip()
code_path = "scripts/course_experiments/compression.py"
pinned = subprocess.check_output(["git", "show", revision + ":" + code_path], cwd=ROOT)
assert sha(pinned) == record["code_sha256"][code_path]
selected_names = ["_parameter_hash", "_cache_text", "_hard_targets", "_fit_text", "_distill_case", "run_distillation"]
tasks = {}
for name, task in record["results"]["tasks"].items():
    hard = task["hard_target_generation"]
    tasks[name] = {
        "teacher_frozen_and_unchanged": task["teacher_frozen_and_unchanged"],
        "teacher_cache": task["teacher_cache"],
        "hard_target_generation": None if hard is None else {k: v for k, v in hard.items() if k != "audit"},
        "circle_example": None if hard is None else next((row for row in hard["audit"] if row["teacher_answer"] == "circle"), None),
        "student_training_records": task["data"]["student_training_records"],
        "runs": {key: {"steps": value["training"]["steps"],
                        "effective_supervised_tokens": value["training"]["effective_supervised_tokens"]}
                 for key, value in task["runs"].items() if "training" in value},
    }
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "device": "讀取文件與紀錄，不執行訓練"},
    "external_sources": external,
    "experiment": {"path": str(record_path.relative_to(ROOT)), "sha256": sha(record_raw),
                   "environment": {key: record[key] for key in ["revision", "python_version", "torch_version", "device", "gpu", "seed", "step_scale", "status", "evidence_status"]},
                   "tasks": tasks, "artifact_manifest": [a for a in record["artifacts"] if "teacher" in a["path"] or "hard-targets" in a["path"]]},
    "recorded_implementation": {"original_revision": record["revision"], "available_revision": revision,
                                "path": code_path, "sha256": sha(pinned),
                                "matches_recorded_code_sha256": True,
                                "functions": functions(pinned, selected_names)},
    "current_implementation": {"path": code_path, "sha256": sha((ROOT / code_path).read_bytes())},
}
with (DESTINATION / "source-inspection.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"external_returncodes": [x["returncode"] for x in external],
                  "recorded_implementation_sha256_matches": True,
                  "teacher_tasks": {key: {"unchanged": value["teacher_frozen_and_unchanged"], "cache": value["teacher_cache"],
                                            "hard_seconds": None if value["hard_target_generation"] is None else value["hard_target_generation"]["seconds"]}
                                    for key, value in tasks.items()}}, ensure_ascii=False))
