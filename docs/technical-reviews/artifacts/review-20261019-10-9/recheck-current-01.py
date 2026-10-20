import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path.cwd()
original_path = Path("docs/course-review-20261019/reviews/rv1019_technical_10-9/technical-10-9-original-01.json")
old_source = Path("docs/course-review-20261019/freeze-01/sources/10.9.md")
new_source = Path("docs/course-review-20261019/freeze-05/sources/10.9.md")
freeze_path = Path("docs/course-review-20261019/freeze-05.json")
original = json.loads(original_path.read_bytes())
assert sha(original_path) == "bcabd5f9ff7e808e3e81294d50e1094d34ebdf9a34ff02d14c5d3439c66ab194"
assert sha(new_source) == "57f7c5f732595b1a088349f0d840038885628aaaccbddceddab51b5e3d530796"
assert sha(freeze_path) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
reuse = []
for item in original["artifacts"]:
    digest = sha(item["path"])
    assert digest == item["sha256"]
    reuse.append({"path": item["path"], "sha256": digest, "unchanged": True})
repository_inputs = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        digest = sha(source["path"])
        assert digest == source["sha256"]
        repository_inputs.append({"path": source["path"], "sha256": digest, "unchanged": True})
record = json.loads(Path("docs/technical-reviews/artifacts/review-20261019-10-9/experiment-inspection.json").read_bytes())
for item in record["inputs"]:
    digest = sha(item["path"])
    assert digest == item["sha256"]
    repository_inputs.append({"path": item["path"], "sha256": digest, "unchanged": True})
diff_command = ["git", "diff", "--no-index", str(old_source), str(new_source)]
difference = subprocess.run(diff_command, capture_output=True, text=True)
assert difference.returncode == 1 and not difference.stderr
helper_command = [".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/10.md#10.9", "--execute"]
helper = subprocess.run(helper_command, capture_output=True, text=True)
helper_receipt = {"command_argv": helper_command, "exit_code": helper.returncode, "stdout": helper.stdout, "stderr": helper.stderr}
assert helper.returncode == 0, helper_receipt
helper_info = json.loads(helper.stdout)
assert helper_info["source_sha256"] == sha(new_source)
run_directory = Path(helper_info["artifact_directory"])
assert (run_directory / "section.md").read_bytes() == new_source.read_bytes()
files = []
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"]:
    path = run_directory / name
    files.append({"name": name, "sha256": sha(path), "content": path.read_text()})
assert sha(run_directory / "fence-1.py") == sha("docs/technical-reviews/artifacts/review-20261019-10-9/fence-1.py")
assert (run_directory / "stdout.txt").read_bytes() == Path("docs/technical-reviews/artifacts/review-20261019-10-9/stdout.txt").read_bytes()
implementation = inspect.getsource(F.normalize)
assert "input.norm(p, dim, keepdim=True)" in implementation
integer = torch.tensor([[0, 3], [2, 0]])
try:
    F.normalize(integer, dim=-1)
except RuntimeError as error:
    integer_result = {"dtype": str(integer.dtype), "error_type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("整數輸入沒有出現預期的範數型別錯誤")
floating = torch.tensor([[0.0, 3.0], [2.0, 0.0]])
float_output = F.normalize(floating, dim=-1)
assert torch.equal(float_output, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
complex_input = torch.tensor([[3.0 + 4.0j, 0.0j]], dtype=torch.complex64)
complex_output = F.normalize(complex_input, dim=-1)
assert not complex_input.is_floating_point()
assert torch.allclose(torch.linalg.vector_norm(complex_output, dim=-1), torch.ones(1))
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-9/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda)},
    "inputs": {"original_report": {"path": str(original_path), "sha256": sha(original_path)}, "current_source": {"path": str(new_source), "sha256": sha(new_source)}, "current_freeze": {"path": str(freeze_path), "sha256": sha(freeze_path)}},
    "reused_artifacts": reuse,
    "repository_inputs": repository_inputs,
    "source_diff": {"command_argv": diff_command, "exit_code": difference.returncode, "stdout": difference.stdout, "stderr": difference.stderr},
    "section_execution": {"receipt": helper_receipt, "files": files, "fence_unchanged": True, "stdout_unchanged": True, "current_section_equals_snapshot": True},
    "normalization_implementation": {"path": inspect.getsourcefile(F.normalize), "file_sha256": sha(inspect.getsourcefile(F.normalize)), "signature": str(inspect.signature(F.normalize)), "inspected_function_sha256": hashlib.sha256(implementation.encode()).hexdigest(), "norm_statement": next(line.strip() for line in implementation.splitlines() if "denom = input.norm(p, dim, keepdim=True).clamp_min(eps)" in line)},
    "dtype_checks": {"integer": integer_result, "floating": {"dtype": str(floating.dtype), "output": float_output.tolist()}, "complex": {"dtype": str(complex_input.dtype), "is_floating_point": complex_input.is_floating_point(), "output_real_imag": [[[float(z.real), float(z.imag)] for z in row] for row in complex_output.tolist()], "output_norms": torch.linalg.vector_norm(complex_output, dim=-1).tolist()}},
    "result": "所有斷言通過；現行小節與凍結版相同，原程式及其輸出不變；浮點成功、整數報錯、複數成功。"
}
print(json.dumps(output, ensure_ascii=False, indent=2))
