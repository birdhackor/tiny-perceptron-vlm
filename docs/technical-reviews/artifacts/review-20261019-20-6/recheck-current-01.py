"""複查來源變動、保留證據與 LoRA 梯度算例；輸出只新增於本目錄。"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.alignment import LoRALinear


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return {"argv": args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_20-6/technical-20-6-original-01.json"
    assert digest(original_path) == "55453a24627b9034f7ba96e1257a0fc314da0994a729d80d928a0f23f037a036"
    original = load(original_path)
    seals = []
    for item in original["sources"] + original["artifacts"]:
        if "path" not in item:
            continue
        actual = digest(ROOT / item["path"])
        changed = actual != item["sha256"]
        assert not changed or item["id"] == "s_prerequisite"
        seals.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": not changed})
    assert digest(HERE / "self-check-01.txt") == "45878218bbdd916f8eab4f612628a4536e4081be39ae9446df0bb5296d085f3c"
    old_page = ROOT / "docs/course-review-20261019/freeze-04/sources/20.6.md"
    current_page = ROOT / "docs/course-review-20261019/freeze-05/sources/20.6.md"
    expected = old_page.read_bytes().replace("完整紀錄（JSON）".encode(), "稽核摘要（JSON）".encode())
    assert current_page.read_bytes() == expected
    assert digest(current_page) == "8a85e317c2ea4ee8e4ec67d5fe8964dc41897cb25c893fd64fc8866ad9ad0e26"
    page_diff = command(["git", "diff", "--no-index", str(old_page.relative_to(ROOT)), str(current_page.relative_to(ROOT))])
    prerequisite_diff = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/chapters/08.md"])
    assert page_diff["exit_code"] == 1 and prerequisite_diff["exit_code"] == 0
    prerequisite_path = ROOT / "course/chapters/08.md"
    assert digest(prerequisite_path) == "860a5f2e3d9788fb22b93f5e7e1ae7df76009987b97f4e6a60a255942401f123"
    text = prerequisite_path.read_text()
    body = re.search(r"(?ms)^## 8\.8 .*?(?=^## |\Z)", text).group()
    fences = re.findall(r"```python\n(.*?)```", body, re.S)
    extracted_class = next(node for node in ast.parse(fences[1]).body if isinstance(node, ast.ClassDef))
    actual_class = next(node for node in ast.parse((ROOT / "tiny_perceptron/alignment.py").read_text()).body if isinstance(node, ast.ClassDef) and node.name == "LoRALinear")
    assert ast.dump(extracted_class) == ast.dump(actual_class)
    stdout = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(stdout):
        exec(compile(fences[0], str(prerequisite_path), "exec"), namespace)
    layer = namespace["layer"]
    assert tuple(layer.base.weight.shape) == (12, 16)
    assert tuple(layer.a.shape) == (2, 16) and tuple(layer.b.shape) == (12, 2)
    assert torch.count_nonzero(layer.a.grad).item() == 0 and torch.count_nonzero(layer.b.grad).item() > 0
    prerequisite = {"stdout": stdout.getvalue(), "class_ast_matches_library": True, "source_sha256": digest(prerequisite_path), "first_a_gradient_zero": True, "first_b_gradient_nonzero": True}
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
    x = torch.stack((torch.ones(16), -torch.ones(16)))
    projected = x @ layer.a.T
    layer(x).sum().backward()
    assert torch.count_nonzero(projected).item() > 0
    assert torch.count_nonzero(layer.b.grad).item() == 0
    cancellation = {"input_shape": list(x.shape), "projected_values": projected.detach().tolist(), "upstream_gradient": "sum() supplies ones to all output positions", "b_gradient_max": float(layer.b.grad.abs().max()), "reason": "opposite input rows cancel in the summed outer products"}
    spec = importlib.util.spec_from_file_location("original_verifier", HERE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    primary = load(HERE / "primary-sources.json")
    config = next(item for item in primary["records"] if item["id"] == "qwen_config")["selected_original_values"]
    examples = module.examples()
    receipts = module.receipts(config)
    run = ROOT / "outputs/reviewer-tools/runs/20261020T101632473528Z"
    execution = {name: load(run / name) for name in ("extraction.json", "execution.json", "environment.json")}
    execution.update({name: (run / name).read_text() for name in ("stdout.txt", "stderr.txt", "fence-1.py")})
    execution["invocation"] = 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.6" --execute'
    assert execution["execution.json"]["exit_code"] == 0
    assert execution["extraction.json"]["source_sha256"] == digest(current_page)
    assert (run / "section.md").read_bytes() == current_page.read_bytes()
    assert execution["fence-1.py"] == load(HERE / "section-execution.json")["fence-1.py"]
    save("section-execution-current-01.json", execution)
    audit_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json"
    audit = load(audit_path)
    assert "history" not in audit["independent_audit"]
    assert "report_sha256" in audit and audit["independent_audit"]["changed_adapter_tensors"] == 112
    record = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + Path(__file__).relative_to(ROOT).as_posix(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "original_report_sha256": digest(original_path), "unchanged_evidence_checks": seals,
        "page_diff": page_diff, "prerequisite_diff": prerequisite_diff,
        "current_section_matches_freeze": True, "python_fence_unchanged": True,
        "prerequisite": prerequisite, "cancellation_example": cancellation,
        "examples": examples, "receipt_checks": receipts,
        "changed_statement_check": {"previous": "完整紀錄（JSON）", "current": "稽核摘要（JSON）", "status": "verified", "summary_sha256": digest(audit_path), "summary_root_keys": list(audit), "basis": "摘要記錄稽核總數、範圍與原始報告指紋；完整原始報告另列於索引。"},
        "scope": "重跑頁面原始程式與原獨立算例，並核對引用前置的定義及梯度條件；未重跑 GPU 長訓練，未讀排除的原始 outputs。",
    }
    save("recheck-current-01.json", record)
    print(json.dumps({"result": "pass", "evidence_files_checked": len(seals), "prerequisite_stdout": prerequisite["stdout"], "page_fence_unchanged": True, "rank_counts": [item["trainable_count"] for item in examples], "cancellation_b_gradient_max": cancellation["b_gradient_max"], "changed_statement_status": "verified"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
