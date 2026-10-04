"""Optional README delta only: real counts/route and corrected GPU environment scope."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "docs/technical-reviews/artifacts"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
from check_technical_reviews import sections, FRONT_MATTER


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


chapters = sorted((ROOT / "course/chapters").glob("*.md"))
numeric_chapters = len([p for p in chapters if p.stem.isdecimal()])
notebooks = len(list((ROOT / "notebooks").rglob("*.ipynb")))
numbered = sum(len(list(sections(p))) for p in chapters + [ROOT / "course" / name for name in FRONT_MATTER])
assert (numeric_chapters, notebooks, numbered) == (20, 261, 287)
plan_path = ROOT / "docs/course-experiments/plan.json"
plan = json.loads(plan_path.read_text())
formal_gpu, formal_cpu = [], []
for item in plan["sequence"]:
    path = ROOT / item["evidence"]
    record = json.loads(path.read_text())
    observed = {"experiment": item["id"], "path": str(path.relative_to(ROOT)), "sha256": sha(path), "plan_device": item["device"], "torch_version": record.get("torch_version"), "gpu": record.get("gpu"), "python": record.get("python_version")}
    if item["device"] == "cuda":
        assert observed["torch_version"] == "2.14.1+cu126" and observed["gpu"] == "NVIDIA L4", observed
        formal_gpu.append(observed)
    else:
        formal_cpu.append(observed)
assert len(formal_gpu) == 29 and [x["experiment"] for x in formal_cpu] == ["simple_models"]
natural_path = ROOT / "docs/natural-assistant/evidence/train/result.json"
natural = json.loads(natural_path.read_text())
assert natural["versions"]["torch"] == "2.8.0+cu128" and natural["gpu_name"] == "NVIDIA L4"
runner_path = ROOT / "scripts/modal_natural.py"
runner = runner_path.read_text()
assert 'modal.Image.debian_slim(python_version="3.12")' in runner
assert '.pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")' in runner
student_path = ROOT / "docs/natural-assistant/STUDENT.md"
student = student_path.read_text()
assert 'torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128' in student
reads = {}
for name in ("README.md", "README_en.md"):
    path = ROOT / name
    text = path.read_text()
    hardware = next(line for line in text.splitlines() if ('正式GPU訓練' in line or 'formal GPU training' in line))
    route = next(line for line in text.splitlines() if '18/18' in line and '1/10' in line)
    assert '2.14.1+cu126' in hardware and '2.8.0' in hardware and '12.8' in hardware and 'STUDENT.md' in hardware
    assert 'Whisper' in route and ('21.3億' in route or '2.13 billion' in route) and ('161萬' in route or '1.61 million' in route)
    snapshot = P / ("natural-final-fact-20.8-training-readme-current-" + name)
    snapshot.write_bytes(path.read_bytes())
    reads[name] = {"sha256": sha(path), "hardware_paragraph_actually_read": hardware, "natural_route_paragraph_actually_read": route, "snapshot": str(snapshot.relative_to(ROOT))}
result = {"scope": "Optional README supplement: counts, natural-route paragraph and hardware environment delta only; no blanket README or macOS/Windows validation claim, no new canonical dependency", "previous_complete_supplement_preserved": "docs/technical-reviews/artifacts/natural-final-fact-20.8-readme-supplement.json", "counts": {"numeric_chapters": numeric_chapters, "notebooks": notebooks, "numbered_sections": numbered}, "readmes": reads, "first19_formal_plan_source": {"path": str(plan_path.relative_to(ROOT)), "sha256": sha(plan_path)}, "first19_actual_formal_gpu_records": formal_gpu, "first19_formal_CPU_exception": formal_cpu, "CPU_supplement_exceptions_not_generalized_to_GPU": ["posttraining", "tool_choice"], "chapter20_actual_original_record": {"path": str(natural_path.relative_to(ROOT)), "sha256": sha(natural_path), "torch": natural["versions"]["torch"], "gpu": natural["gpu_name"]}, "chapter20_actual_runner_source": {"path": str(runner_path.relative_to(ROOT)), "sha256": sha(runner_path), "python_version": "3.12", "torch_cuda_package": "2.8.0/cu128"}, "student_install_source": {"path": str(student_path.relative_to(ROOT)), "sha256": sha(student_path)}, "matrix_crosscheck": "Previous personally audited exact2,127,532,032/1,605,632 and originalsemantic denominators18/18 synthetic vs1/10 external remain unchanged. Whisper -> samechat actualcore read.", "hardware_scope_issue_resolution": "Root added正式GPU訓練/formal GPU training, so first19 CPU formal simple_models and CPU supporting experiments are not wrongly described as L4 runs; both corrected paragraphs personally reread.", "no_installs_or_GPU_executed": True}
(P / "natural-final-fact-20.8-training-readme-supplement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"counts": result["counts"], "formal_gpu_records_verified": len(formal_gpu), "formal_cpu_exception": "simple_models", "README_GPU_scope_pass": True}))
