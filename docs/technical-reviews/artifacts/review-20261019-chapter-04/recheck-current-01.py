"""以新檔保存模型重跑、依賴文字差異及 API 核對，不改寫舊證據。"""

import contextlib
import hashlib
import html
import io
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-04/technical-chapter-04-original-01.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, content):
    with path.open("x") as stream:
        stream.write(content)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def section(text, identifier):
    match = re.search(r"^## " + re.escape(identifier) + r" .*$", text, re.MULTILINE)
    assert match is not None, "找不到要核對的小節"
    following = re.search(r"^## ", text[match.end():], re.MULTILINE)
    end = match.end() + following.start() if following else len(text)
    return text[match.start():end]


def run_blocks(text, path):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        for code in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            exec(compile(code, path, "exec"), namespace)
    return output.getvalue(), namespace


assert digest(ORIGINAL) == "25ad1b3b52ddeff2d346fbf5f036467b27dc7c675693f7f583351365d5fb9806"
original = json.loads(ORIGINAL.read_text())
old_files = {ORIGINAL: digest(ORIGINAL)}
for artifact in original["artifacts"]:
    path = ROOT / artifact["path"]
    assert digest(path) == artifact["sha256"], "舊證據的雜湊已變動"
    old_files[path] = digest(path)
old_self_check = BASE / "self-check.txt"
old_files[old_self_check] = digest(old_self_check)

expected_sources = {
    "course/chapters/03.md": "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09",
    "course/chapters/14.md": "40448837dfb1418b4fdfdad4544e133e6554bb80831ae2a2802ef6ab3ffc7601",
}
diffs = []
section_comparisons = []
for path, expected in expected_sources.items():
    assert digest(ROOT / path) == expected, "現行檔案與派工指紋不符"
    command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", path]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    before = git("show", "5ef56bc756f3:" + path)
    recorded = next(source for source in original["sources"] if source.get("path") == path)
    assert hashlib.sha256(before.encode()).hexdigest() == recorded["sha256"], "差異起點與原審閱版本不符"
    assert git("show", "3af7f13ef959:" + path) == (ROOT / path).read_text(), "差異終點與現行版本不符"
    diffs.append({"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    identifiers = ["3.2", "3.6", "3.7"] if path.endswith("03.md") else ["14.11"]
    for identifier in identifiers:
        old_text = section(before, identifier)
        new_text = section((ROOT / path).read_text(), identifier)
        assert old_text == new_text, "原先引用的依據段落已有實質變動"
        section_comparisons.append({"path": path, "selector": identifier, "unchanged": True, "sha256": hashlib.sha256(new_text.encode()).hexdigest()})

model_code = (BASE / "verify_model.py").read_text()
model_code = model_code.replace("freeze-04.json", "freeze-05.json")
model_code = model_code.replace("a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba")
model_code = model_code.replace("repository-inspection.json", "repository-inspection-current-01.json")
model_code = model_code.replace('(ARTIFACTS / "repository-inspection-current-01.json").write_text(', 'write_new(ARTIFACTS / "repository-inspection-current-01.json", ')
model_code = model_code.replace('/verify_model.py"', '/recheck-current-01.py"')
model_output = io.StringIO()
with contextlib.redirect_stdout(model_output):
    exec(compile(model_code, str(BASE / "verify_model.py"), "exec"), {"__file__": str(BASE / "verify_model.py"), "__name__": "__main__", "write_new": write_new})
write_new(BASE / "model-verification-current-01.txt", model_output.getvalue())
model_run = json.loads(model_output.getvalue())

text03 = (ROOT / "course/chapters/03.md").read_text()
text14 = (ROOT / "course/chapters/14.md").read_text()
scale_stdout, _ = run_blocks(section(text03, "3.5"), "course/chapters/03.md")
norm_stdout, norm_namespace = run_blocks(section(text14, "14.2"), "course/chapters/14.md")
assert torch.allclose(norm_namespace["norm"](norm_namespace["x"]), norm_namespace["rms"]), "RMSNorm 與手算超出容忍差"

scores = torch.tensor([-4.0, *([0.0] * 7), 4.0])
assert scores.std().item() == 2.0 and (scores / 2).std().item() == 1.0
sampling = {
    "chapter_stdout": scale_stdout,
    "possible_finite_sample_scores": scores.tolist(),
    "sample_mean": scores.mean().item(),
    "sample_std": scores.std().item(),
    "scaled_sample_std": (scores / 2).std().item(),
    "derivation": "D 個獨立 ±1 乘積各有平均 0、變異數 1，相加的母體變異數為 D。有限樣本估計可能偏離母體值，但不必偏離。D=4 時可出現分數 [-4,0,0,0,0,0,0,0,4]：平均 0，平方和 32，樣本變異數為 32/(9-1)=4，標準差為 2；除以 sqrt(4)=2 後標準差恰為 1。這些分數皆是四個 ±1 乘積可產生、機率非零的結果。"
}

positive = torch.tensor([0.0])
negative = torch.tensor([-0.0])
first = torch.tensor([1.0])
nearby = torch.nextafter(first, torch.tensor([float("inf")]))
equality = {
    "signed_zero_equal": torch.equal(positive, negative),
    "signed_zero_storage_bits_equal": torch.equal(positive.view(torch.int32), negative.view(torch.int32)),
    "different_dtype_same_values_equal": torch.equal(torch.tensor([1], dtype=torch.int32), torch.tensor([1.0])),
    "nearby_values_equal": torch.equal(first, nearby),
    "nearby_values_allclose": torch.allclose(first, nearby),
    "nan_self_equal": torch.equal(torch.tensor([float("nan")]), torch.tensor([float("nan")])),
    "chapter_stdout": norm_stdout,
    "rms_max_abs_difference": (norm_namespace["norm"](norm_namespace["x"]) - norm_namespace["rms"]).abs().max().item(),
}
assert equality["signed_zero_equal"] and not equality["signed_zero_storage_bits_equal"]
assert equality["different_dtype_same_values_equal"] and not equality["nearby_values_equal"] and equality["nearby_values_allclose"]
assert not equality["nan_self_equal"]

from tiny_perceptron.model import ModelConfig, TinyLM
configuration = {}
for name, value in [("norm", "rms"), ("rotary", True), ("activation", "relu2"), ("activation", "swiglu"), ("tied", True)]:
    model = TinyLM(ModelConfig(width=8, **{name: value}))
    key = f"{name}={value}"
    if name == "norm":
        matched = all(type(block.norm1).__name__ == "RMSNorm" and type(block.norm2).__name__ == "RMSNorm" for block in model.blocks) and type(model.final_norm).__name__ == "RMSNorm"
    elif name == "rotary":
        matched = model.position is None and all(block.attention.rotary for block in model.blocks)
    elif name == "activation":
        matched = all(block.ffn.activation == value for block in model.blocks)
    else:
        matched = model.output.weight is model.embedding.weight
    assert matched, "設定沒有換上對應零件"
    assert model(torch.tensor([[1, 2, 3]]))["logits"].shape == (1, 3, 264)
    configuration[key] = {"matching_component_verified": matched, "forward_shape": [1, 3, 264]}

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)

url = "https://docs.pytorch.org/docs/2.14/generated/torch.equal.html"
curl_command = ["curl", "-fsSL", "--max-time", "30", url]
curl = subprocess.run(curl_command, capture_output=True, check=True)
parser = Text()
parser.feed(curl.stdout.decode())
normalized = " ".join(html.unescape(" ".join(parser.parts)).split())
excerpt = "True if two tensors have the same size and elements, False otherwise."
assert excerpt in normalized, "官方文件中找不到 API 契約"
assert excerpt in " ".join(torch.equal.__doc__.replace("``", "").split()), "安裝版本的 API 契約不同"
source_receipt = {
    "url": url, "version": "PyTorch 2.14 官方文件；本機 torch " + str(torch.__version__),
    "accessed_on": datetime.now(timezone.utc).astimezone().date().isoformat(),
    "locator": "torch.equal 的 API 說明與緊接的 Note、Examples",
    "command": curl_command, "exit_code": curl.returncode, "stderr": curl.stderr.decode(),
    "response_sha256": hashlib.sha256(curl.stdout).hexdigest(),
    "excerpt": excerpt, "excerpt_present_in_installed_docstring": True, "full_text_saved": False,
}
assert all(digest(path) == previous for path, previous in old_files.items()), "核對過程改動了舊報告或證據"
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --offline python docs/technical-reviews/artifacts/review-20261019-chapter-04/recheck-current-01.py",
    "environment": model_run["environment"], "original_issues": original["issues"],
    "original_file_integrity": [{"path": path.relative_to(ROOT).as_posix(), "sha256": value, "unchanged": True} for path, value in old_files.items()],
    "dependency_diffs": diffs, "unchanged_dependency_sections": section_comparisons,
    "model_reexecution": {"path": (BASE / "model-verification-current-01.txt").relative_to(ROOT).as_posix(), "sha256": digest(BASE / "model-verification-current-01.txt")},
    "model_reexecution_adjustments": ["使用新凍結 manifest 與其已核對指紋", "相同核對程式在記憶體改為輸出新的 repository-inspection 檔名並使用排他寫入", "執行方法及模型核對不變"],
    "sampling_statement": sampling, "equality_statement": equality,
    "conditional_library_configuration_statement": configuration,
    "original_official_docs": source_receipt,
    "result": "原報告與舊證據完整；依賴段落及模型重跑通過；三處改寫均有依據。",
}, ensure_ascii=False, indent=2))
