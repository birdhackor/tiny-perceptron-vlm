"""Second W.1 callback: new cumulative raw outputs and an isolated real export."""

import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_1"
ORIGINAL = REPORT_DIR / "technical-original-v1.json"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json"
CANONICAL = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix06.json"
sys.path.insert(0, str(ROOT))

from scripts.build_course import BOOTSTRAP, notebook
from scripts.export_course import checked_notebook, output_html, section_footer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
    result = {"command": shlex.join(args), "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    assert completed.returncode == 0, result
    return result


started_at = datetime.now(UTC).isoformat()
assert digest(ORIGINAL) == "18525e63a0d34a51e9acc1eab7938842138b672ba70f33b03981ceb7cebe5175"
original = json.loads(ORIGINAL.read_bytes())
freeze = json.loads(FREEZE.read_bytes())
canonical = json.loads(CANONICAL.read_bytes())
assert freeze["canonical_source_freeze"] == {"path": str(CANONICAL.relative_to(ROOT)), "sha256": digest(CANONICAL)}
page = next(p for p in freeze["pages"] if p["page_id"] == "W.1")
parent = next(p for p in canonical["pages"] if p["page_id"] == "first-steps")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
old_freeze_path = ROOT / original["source_freeze"]["path"]
old_freeze = json.loads(old_freeze_path.read_bytes())
old_page = next(p for p in old_freeze["pages"] if p["page_id"] == "W.1")
assert raw == (ROOT / old_page["snapshot"]).read_bytes()
v3_path = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3.json"
v3 = next(p for p in json.loads(v3_path.read_bytes())["pages"] if p["page_id"] == "W.1")
assert raw == (ROOT / v3["snapshot"]).read_bytes()
parent_bytes = (ROOT / parent["snapshot"]).read_bytes()
assert hashlib.sha256(parent_bytes).hexdigest() == parent["source_sha256"]
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", parent_bytes, re.M | re.S)[0] == raw
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ROOT / page["source"]).read_bytes(), re.M | re.S)[0] == raw
assert page["figures_sha256"] == original["figure_sha256"] == {}
assert not re.findall(rb"!\[[^\]]*\]\([^)]*\)", raw)

dependencies = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(ROOT / source["path"])
        dependencies.append({"source_id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": actual, "identical": actual == source["sha256"]})
        assert dependencies[-1]["identical"], dependencies[-1]
preserved_artifacts = []
for artifact in original["artifacts"]:
    actual = digest(ROOT / artifact["path"])
    assert actual == artifact["sha256"], artifact["path"]
    preserved_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual})
official_sources = json.loads((ARTIFACTS / "official-source-inspection.json").read_bytes())
for receipt in official_sources:
    assert digest(Path(receipt["raw_path"])) == receipt["raw_sha256"], receipt["id"]

import torch
import nbformat
from nbclient import NotebookClient

environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "device": "cpu", **{p: importlib.metadata.version(p) for p in ["jupyterlab", "ipykernel", "nbclient"]}}
assert torch.version.cuda is None
check_env = command([".venv/bin/python", "scripts/check_env.py"])
assert "compute         OK（cpu）" in check_env["stdout"]
assert "未安裝" not in check_env["stdout"]
lab = command([".venv/bin/python", "-m", "jupyterlab", "notebooks", "--show-config-json"])
assert json.loads(lab["stdout"])["ServerApp"]["root_dir"] == str(ROOT / "notebooks")
kernel = json.loads((ROOT / ".venv/share/jupyter/kernels/tiny-perceptron/kernel.json").read_bytes())
assert kernel["display_name"] == "Tiny Perceptron"
assert Path(kernel["argv"][0]) == ROOT / ".venv/bin/python"

nb_path = ROOT / "notebooks/01/1.1.ipynb"
nb = json.loads(nb_path.read_bytes())
setup = next(c for c in nb["cells"] if c.get("metadata", {}).get("course_setup"))
assert "".join(setup["source"]) == BOOTSTRAP.rstrip("\n")
chapter = (ROOT / "course/chapters/01.md").read_text()
section11 = re.search(r"^## 1\.1 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
inline = re.search(r"```python\n(.*?)```", section11, re.S)[1].strip()
code = next(c for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {}))
assert "".join(code["source"]).strip() == inline
exercise = []
for text in ["貓看狗，狗看貓。", "鳥看狗，狗看鳥。"]:
    probe = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("".join(setup["source"])),
        nbformat.v4.new_code_cell(inline.replace("貓看狗，狗看貓。", text)),
        nbformat.v4.new_code_cell('print(root); print(torch.empty(0).device); retained = "state"'),
        nbformat.v4.new_code_cell('print(retained); print(restored)'),
    ])
    NotebookClient(probe, timeout=45, kernel_name="tiny-perceptron", resources={"metadata": {"path": str(ROOT / "notebooks/01")}}).execute()
    texts = ["".join(o.get("text", "") for o in c.outputs if o.output_type == "stream") for c in probe.cells]
    assert text in texts[1] and text in texts[3]
    assert str(ROOT) in texts[2] and "cpu" in texts[2]
    assert "state" in texts[3]
    exercise.append({"input": text, "outputs": texts, "new_kernel": True, "execution_counts": [c.execution_count for c in probe.cells]})

manifest_path = ARTIFACTS / "cpu-output-dependency-manifest.json"
old_manifest = json.loads(manifest_path.read_bytes())
bundle_receipt_path = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix06-cumulative.json"
bundle_receipt = json.loads(bundle_receipt_path.read_bytes())
by_id = {row["lesson"]: row for row in bundle_receipt["records"]}
manifest = []
for row in old_manifest:
    current_row = by_id[row["lesson_id"]]
    manifest.append({**row, "notebook_sha256": current_row["notebook_sha256"], "executed_copy": str(Path(current_row["bundle_path"]).relative_to(ROOT)), "executed_sha256": current_row["executed_sha256"]})
isolated_receipt_path = ARTIFACTS / "isolated-export-current-v1.json"
isolated_receipt = json.loads(isolated_receipt_path.read_bytes())
assert isolated_receipt["exit_code"] == 0
assert isolated_receipt["bundle_verified"] == isolated_receipt["isolated_downloads_identical"] == 331
assert isolated_receipt["author_receipt"]["sha256"] == digest(bundle_receipt_path)
isolated_site = Path(isolated_receipt["isolated_root"]) / "outputs/site"
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
assert {r["lesson_id"] for r in manifest} == {r["id"] for r in index}
raw_drift, notebook_drift, download_drift = [], [], []
teaching_cells = displayed_cells = code_sections = 0
for receipt in manifest:
    notebook_path = ROOT / receipt["notebook"]
    execution_path = ROOT / receipt["executed_copy"]
    current = json.loads(notebook_path.read_bytes())
    executed = checked_notebook(current, execution_path)
    if digest(notebook_path) != receipt["notebook_sha256"]:
        notebook_drift.append(receipt["lesson_id"])
    if digest(execution_path) != receipt["executed_sha256"]:
        raw_drift.append(receipt["lesson_id"])
    code_cells = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    displays = sum(bool(output_html(c.get("outputs", []))) for c in code_cells)
    if code_cells:
        assert displays > 0, receipt["lesson_id"]
        code_sections += 1
    teaching_cells += len(code_cells)
    displayed_cells += displays
    copied = isolated_site / receipt["notebook"]
    if copied.read_bytes() != notebook_path.read_bytes():
        download_drift.append(receipt["lesson_id"])
assert not notebook_drift and not raw_drift and not download_drift

footer_checks = []
for label, notebook_value, flag in [
    ("python_with_outputs", nb, True),
    ("python_without_outputs", nb, False),
    ("terminal_only", notebook("X.1", "terminal", "```bash\npython scripts/check_env.py\n```", ROOT / "course/first-steps.md"), True),
    ("prose_only", notebook("X.1", "prose", "plain", ROOT / "course/first-steps.md"), True),
]:
    footer = section_footer({"notebook": "notebooks/01/1.1.ipynb"}, notebook_value, "W1-current-v4", flag)
    assert 'href="notebooks/01/1.1.ipynb" download' in footer
    assert "/blob/W1-current-v4/notebooks/01/1.1.ipynb" in footer
    assert ("CPU 實際輸出" in footer) == (label == "python_with_outputs")
    if label in {"terminal_only", "prose_only"}:
        assert "本節沒有要在 Notebook 執行的 Python 程式" in footer
    else:
        assert "notebooks/" in footer and "雲端環境未經本課實測" in footer
    footer_checks.append({"case": label, "sha256": hashlib.sha256(footer.encode()).hexdigest(), "result": "expected download/Colab links and scope-specific explanatory text"})
negative_checks = []
for case in ["source_drift", "not_executed", "error_output"]:
    modified = copy.deepcopy(json.loads((ROOT / manifest[0]["executed_copy"]).read_bytes()))
    target_cell = next(c for c in modified["cells"] if c["cell_type"] == "code")
    if case == "source_drift":
        target_cell["source"] = ["print('different')"]
    elif case == "not_executed":
        target_cell["execution_count"] = None
    else:
        target_cell["outputs"] = [{"output_type": "error", "ename": "Synthetic", "evalue": "probe", "traceback": []}]
    bad = Path("/workspace/work/omni-technical-w-1-v1/current-negative.ipynb")
    bad.write_text(json.dumps(modified))
    try:
        checked_notebook(json.loads((ROOT / manifest[0]["notebook"]).read_bytes()), bad)
    except ValueError as e:
        negative_checks.append({"case": case, "rejected": True, "error": str(e)})
    else:
        raise AssertionError(case)

claim_notes = {
    "c01": "重新讀靜態輸出轉HTML分支；未增加安裝或kernel閱讀前提。",
    "c02": "重新核新的fix06累積bundle331份source/code/output/manifest指向及319個含程式節的非空輸出；9.11的新run已完成且來源相符，隔離export實際呈現新輸出。",
    "c03": "W.1的Notebook定義逐字不變；原始Jupyter官方定義與指紋仍是同版。",
    "c04": "共享暫存9.11舊下載未被修改；已用新bundle與未改的exporter做Git外真建置，331/331新下載與當前Notebook逐byte相同，並重新核頁尾連結。",
    "c05": "重讀準備格向上找pyproject及共用程式分支，並由notebooks/01新kernel真跑確認root。",
    "c06": "Colab帳號/VM限制措辭未改；原官方FAQ同版且不增加Colab實測主張。",
    "c07": "重讀Colab clone、skip-smudge、pip與check=True；未冒稱跑雲端。",
    "c08": "uv/Python下載措辭、lock與project依賴均未變；原安裝收據仍完整。",
    "c09": "Git命令與目錄說明逐字不變；官方clone契約及原Git版本實跑收據仍支持。",
    "c10": "skip-smudge命令、訓練LFS attributes與官方smudge版本均未變；本次probe不讀資料包。",
    "c11": "PowerShell env設定逐字不變，官方語法及未跑Windows界線沿用。",
    "c12": "260敘述不變，原數值檢查及官方MAX_PATH/248定位指紋不變，未加Windows實測。",
    "c13": "longpaths/global命令與官方Git設定分支不變，保持目前使用者範圍解讀。",
    "c14": "--frozen、uv.lock精確bytes與原隔離安裝證據皆保持。",
    "c15": "CPU extra/index/lock不變，本次torch build仍為CPU且compute實跑成功。",
    "c16": "notebook group不變，本次Jupyter與新kernel實際可用。",
    "c17": ".venv說明未改，原預設路徑dry-run/隔離安裝收據完整。",
    "c18": "activation字串不變，原Bash PATH實跑及官方Windows表格仍支持，未冒稱跨平台實測。",
    "c19": "再次實跑不啟用的.venv Python及-m jupyterlab路線，root_dir仍為notebooks。",
    "c20": "execution policy的診斷條件與非啟用替代路線未改，維持官方文件範圍。",
    "c21": "重讀compute檢查關鍵行，再次真跑compute OK(cpu)且無未安裝套件。",
    "c22": "現存kernelspec仍指向同一.venv Python/display-name，並由它新開kernel執行。",
    "c23": "本次Jupyter配置解析仍得到repo/notebooks及/lab，沒有擴張視覺實測結論。",
    "c24": "兩個新kernel各自準備，格間retained名稱可讀，維持kernel狀態比喻。",
    "c25": "官方package/快捷鍵證據版本不變，原Shift Enter設定及本次cell執行皆有效。",
    "c26": "同一kernel後格讀先格狀態，本次再次執行，未把修改文字誤當自動重算。",
    "c27": "原重啟再runCells官方原始碼不變，本次兩個獨立kernel重現輸出。",
    "c28": "重新抽當前1.1真正內嵌程式，與Notebook相同，原句與鳥句各在新kernel成功還原。",
    "c29": "本次實際setup後再新增probe格，root、torch CPU及格間名稱皆可用；不延伸成W.2-W.7審閱。",
    "c30": "縮排警語逐字不變；官方Python3.13.5語句分組定位與指紋不變。",
    "c31": "本次CPU probe無訓練資料讀取；必要1.1及LFS attributes未變。",
    "c32": "原Git LFS版本與隔離local install收據完整，官方version/install指紋及教材操作未變。",
    "c33": "本次CPU check_env與1.1仍成功，下載/Colab/正式訓練界線未被改成離線或品質保證。",
}
assert set(claim_notes) == {c["id"] for c in original["claims"]}
result = {
    "started_at": started_at,
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": original["reviewer_task"],
    "lesson_id": "W.1",
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(ORIGINAL)},
    "original_source_freeze": original["source_freeze"],
    "current_source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "current_canonical_source_freeze": {"path": str(CANONICAL.relative_to(ROOT)), "sha256": digest(CANONICAL)},
    "source_equivalence": {"snapshot": page["snapshot"], "source_sha256": page["source_sha256"], "same_bytes_as_original": True, "same_bytes_as_v3": True, "exact_slice_of_current_canonical_parent": True, "current_parent_sha256": parent["source_sha256"], "figure_sha256": {}, "figure_applicability": "W.1 no figures; no visual figure inspection claimed"},
    "environment": environment,
    "repository_dependencies": dependencies,
    "original_artifacts_preserved": preserved_artifacts,
    "official_original_cache_revalidated": len(official_sources),
    "check_env": check_env,
    "jupyter_directory_parse": lab,
    "kernel_argv": kernel["argv"],
    "exercise_new_kernel_runs": exercise,
    "cpu_output_and_download_recheck": {"original_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": digest(manifest_path)}, "current_author_bundle_receipt": {"path": str(bundle_receipt_path.relative_to(ROOT)), "sha256": digest(bundle_receipt_path)}, "isolated_export_receipt": {"path": str(isolated_receipt_path.relative_to(ROOT)), "sha256": digest(isolated_receipt_path)}, "download_site_used": str(isolated_site), "stale_shared_stage_retained": isolated_receipt["stale_shared_downloads_preserved"], "notebooks": len(manifest), "matched_executed_copies": len(manifest), "code_sections": code_sections, "teaching_cells": teaching_cells, "displayed_cells": displayed_cells, "notebook_drift": notebook_drift, "raw_execution_drift": raw_drift, "download_drift": download_drift},
    "footer_cases": footer_checks,
    "exporter_negative_checks": negative_checks,
    "claims_rechecked": [{"claim_id": key, "status": "verified", "inspection_note": value, "scope": "Original claim scope retained; only W.1 assigned."} for key, value in claim_notes.items()],
    "issues": [{"id": "w1-current-dependency-drift", "status": "resolved", "details": "The earlier true failed gate/observation is retained. New 9.11 kernel data and the exact-source331 cumulative bundle pass independent source/code/output checks; an unmodified isolated export produces331 matching downloads. The shared stale9.11 download is explicitly retained and not claimed fixed or deployed."}],
    "verdict": "pass",
    "limitations": ["Only W.1 full slice and necessary program/output/Notebook/footer dependencies; no fresh technical review of W.2-W.7 or the whole first-steps page.", "No new Windows, macOS, Colab or full-book fresh execution; the original platform and empirical limitations remain.", "Two reader task parameters disclose identity exclusions only; no claim that the new partial auxiliary reader completed a page proof."]
}
target = ARTIFACTS / "recheck-current-v2.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": digest(target), "verdict": result["verdict"], "claim_count": len(claim_notes), "cpu_recheck": result["cpu_output_and_download_recheck"], "original_unchanged": digest(ORIGINAL)}, ensure_ascii=False, indent=2))
