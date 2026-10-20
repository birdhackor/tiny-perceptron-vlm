"""核對凍結教材的公開配置、操作契約與既有量測紀錄。"""

import hashlib
import atexit
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
PIN = "ee17b8871cfe2ba9807b796dd11f339b5dae3b5c"
EVIDENCE = ROOT / "docs/natural-assistant/evidence/v4-research/student-selected-public-ui"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
result = {"environment": {"python": sys.version, "platform": platform.platform()}, "commands": []}
atexit.register(lambda: print(json.dumps(result, ensure_ascii=False, indent=2)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(args, *, expected=0, env=None):
    process = subprocess.run(args, capture_output=True, text=True, env=env)
    record = {"command": args, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    result["commands"].append(record)
    assert process.returncode == expected, "指令結束狀態與預期不同"
    return process.stdout


def read_json(path):
    return json.loads(Path(path).read_text())


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


def original(identifier, url, version, locator, excerpt):
    args = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "35", url]
    process = subprocess.run(args, capture_output=True)
    assert process.returncode == 0, "官方原文讀取失敗"
    raw = process.stdout
    parser = PlainText()
    parser.feed(raw.decode())
    plain = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    assert excerpt in plain, "官方原文未找到指定摘錄"
    record = {
        "id": identifier, "url": url, "version": version, "accessed_on": "2026-10-20",
        "locator": locator, "excerpt": excerpt, "checked_original": True,
        "full_response_sha256": digest(raw), "response_bytes": len(raw),
        "command": args, "returncode": process.returncode,
    }
    result.setdefault("official_sources", []).append(record)


for package in ["torch", "Pillow", "soundfile", "numpy", "pytest", "huggingface-hub"]:
    result["environment"][package] = importlib.metadata.version(package)
result["head"] = command(["git", "rev-parse", "HEAD"]).strip()
result["git_version"] = command(["git", "--version"]).strip()
result["source_hashes"] = {}
paths = [
    "docs/course-review-20261019/freeze-01/sources/20.2.md",
    "docs/course-review-20261019/freeze-04.json",
    "scripts/fetch_natural_release.py", "tiny_perceptron/natural_ui.py",
    "tiny_perceptron/natural_assistant.py", "requirements-natural.txt",
    "docs/natural-assistant/v4/public-release.json", "notebooks/20/20.2.ipynb",
    "tests/test_natural_ui.py", "tests/test_natural_release.py", "tests/test_natural_base_release.py",
]
for relative in paths:
    result["source_hashes"][relative] = digest((ROOT / relative).read_bytes())
source = (ROOT / paths[0]).read_text()
assert result["source_hashes"][paths[0]] == "8bb8ce4e5ab32c27337ce2df1081dbe85bdc817bffaee1501847900af8c688dd"
assert result["source_hashes"][paths[1]] == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
result["source_structure"] = {"python_fences": len(re.findall(r"^```python", source, re.M)), "svg_references": re.findall(r"[^\s()]+\.svg", source)}
notebook = read_json(ROOT / "notebooks/20/20.2.ipynb")
result["notebook"] = {"cell_types": [cell["cell_type"] for cell in notebook["cells"]]}
assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"])
result["pinned_files"] = {}
for relative in paths[2:7]:
    pinned = subprocess.run(["git", "show", f"{PIN}:{relative}"], capture_output=True, check=True).stdout
    result["pinned_files"][relative] = {"sha256": digest(pinned), "same_as_worktree": pinned == (ROOT / relative).read_bytes()}
    assert result["pinned_files"][relative]["same_as_worktree"]

manifest = read_json(ROOT / "docs/natural-assistant/v4/public-release.json")
spec = importlib.util.spec_from_file_location("reviewed_fetch", ROOT / "scripts/fetch_natural_release.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
fetch.validate_manifest(manifest)
result["manifest"] = manifest
result["requirements_consistent"] = digest((ROOT / "requirements-natural.txt").read_bytes()) == manifest["requirements_sha256"]
assert result["requirements_consistent"]
result["student_options"] = vars(fetch.student_options(manifest, ARTIFACT, device="cpu"))
result["student_options"] = {key: str(value) for key, value in result["student_options"].items()}
command([sys.executable, "scripts/fetch_natural_release.py", "--manifest", "docs/natural-assistant/v4/public-release.json", "--list"])
command([sys.executable, "scripts/fetch_natural_release.py", "--manifest", "docs/natural-assistant/v4/public-release.json", "--device", "mps", "--list"], expected=2)

with tempfile.TemporaryDirectory(prefix="verification-", dir=ARTIFACT) as temporary:
    temporary = Path(temporary)
    def anonymous_download(repo, path, *, revision, token):
        assert token is False
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
        args = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "35", url]
        process = subprocess.run(args, capture_output=True)
        assert process.returncode == 0, "公開文件匿名下載失敗"
        target = temporary / Path(path).name
        target.write_bytes(process.stdout)
        result.setdefault("anonymous_files", []).append({"url": url, "command": args, "returncode": process.returncode, "bytes": len(process.stdout), "sha256": digest(process.stdout)})
        return target

    destination = temporary / "release"
    fetch.fetch_release(manifest, destination, downloader=anonymous_download)
    command([sys.executable, "scripts/fetch_natural_release.py", "--manifest", "docs/natural-assistant/v4/public-release.json", "--output", str(destination), "--verify"])
    result["download_verified"] = True
    result["download_contents"] = sorted(path.name for path in destination.iterdir())
    try:
        fetch.fetch_release(manifest, destination, downloader=anonymous_download)
    except ValueError as error:
        result["existing_directory_rejected"] = str(error)
    else:
        raise AssertionError("既有下載目錄未被拒絕")
    (destination / "README.md").write_bytes(b"changed")
    try:
        fetch.verify_release(manifest, destination)
    except ValueError as error:
        result["corruption_rejected"] = str(error)
    else:
        raise AssertionError("內容損壞未被拒絕")
    tests = ["tests/test_natural_release.py", "tests/test_natural_base_release.py", "tests/test_natural_ui.py"]
    command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(temporary / "pytest"), *tests])

result["historical_inputs"] = {}
historical = {}
for relative in ["actual-ui/report.json", "actual-ui/browser/report.json", "actual-ui/disk-observation.json", "actual-ui/observer.jsonl", "closure-environment.json", "executed-harness/ui_observer.py.source.txt", "executed-harness/run_ui_smoke.py.source.txt"]:
    raw = (EVIDENCE / relative).read_bytes()
    result["historical_inputs"][relative] = {"path": str((EVIDENCE / relative).relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}
    if relative.endswith(".json"):
        historical[relative] = json.loads(raw)
trial = historical["actual-ui/report.json"]
browser = historical["actual-ui/browser/report.json"]
disk = historical["actual-ui/disk-observation.json"]
events = [json.loads(line) for line in (EVIDENCE / "actual-ui/observer.jsonl").read_text().splitlines()]
resource = next(event for event in events if event["kind"] == "process_resources")
assert resource == trial["server_cpu_resources"]
assert all((ROOT / item["path"]).read_bytes() == subprocess.run(["git", "show", f"{PIN}:" + item["path"]], capture_output=True, check=True).stdout for item in trial["sources_before"] if item["path"] in paths[2:7])
result["historical_observation"] = {
    "model_server_environment": historical["closure-environment.json"]["model_server"],
    "dependencies": trial["dependencies"], "command": trial["command"],
    "cache_bytes": disk["official_cache"]["unique_regular_file_bytes"],
    "cache_gib": disk["official_cache"]["unique_regular_file_bytes"] / 2**30,
    "server_max_rss_kib": resource["maximum_resident_set_kib"],
    "server_max_rss_gib": resource["maximum_resident_set_kib"] / 2**20,
    "ready_seconds": trial["ready_seconds"], "chat_seconds": browser["chat_seconds"],
    "torch_threads": trial["torch_threads"], "torch_interop_threads": trial["torch_interop_threads"],
    "model_load_counts": trial["model_load_counts"], "reset_states": trial["reset_states"],
    "scope": trial["scope"], "offline": trial["offline"],
}
assert round(result["historical_observation"]["cache_gib"], 2) == 5.48
assert round(result["historical_observation"]["server_max_rss_gib"], 2) == 12.87
assert round(trial["ready_seconds"], 2) == 24.24
assert [round(value, 2) for value in browser["chat_seconds"]] == [3.42, 42.87, 36.58, 37.11]
assert trial["offline"] and trial["status"] == "passed"
result["observer_summary"] = {kind: sum(event["kind"] == kind for event in events) for kind in {event["kind"] for event in events}}
original("venv", "https://docs.python.org/3.12/library/venv.html", "Python 3.12 documentation", "Creating virtual environments / How venvs work", "You don’t specifically need to activate a virtual environment")
original("ensurepip", "https://docs.python.org/3.12/library/ensurepip.html", "Python 3.12 documentation", "Introduction", "pip is an independent project with its own release cycle")
original("installing", "https://docs.python.org/3.12/installing/index.html", "Python 3.12 documentation", "Key terms", "it is included by default with the Python binary installers")
original("pytorch", "https://pytorch.org/get-started/previous-versions/", "PyTorch 2.8.0 installation block", "v2.8.0 / Linux and Windows / CPU only", "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu")
original("clone", "https://git-scm.com/docs/git-clone", "Git online manual, accessed 2026-10-20", "--no-checkout", "Do not checkout HEAD after the clone is complete.")
original("checkout", "https://git-scm.com/docs/git-checkout", "Git online manual, accessed 2026-10-20", "--detach", "it points HEAD at the commit ID.")
original("powershell", "https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.5", "PowerShell 7.5", "Use the variable syntax", "$Env:<variable-name> = \"<new-value>\"")
original("git-lfs", "https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.0/docs/man/git-lfs-config.adoc", "git-lfs v3.7.0", "GIT_LFS_SKIP_SMUDGE", "GIT_LFS_SKIP_SMUDGE")
original("hub", "https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/src/huggingface_hub/file_download.py", "huggingface_hub v0.36.2", "hf_hub_download", "hf_hub_download")
result["limitations"] = ["未重新下載官方模型權重或執行成熟模型推論；量測主張核對保存的原始執行紀錄。", "本機為 macOS Python 3.13；本次離線介面測試只支持未變更程式的機制，不是 Python 3.12 Linux 安裝重演。", "未在 Windows、WSL、NVIDIA GPU 或 Apple MPS 實測。"]
