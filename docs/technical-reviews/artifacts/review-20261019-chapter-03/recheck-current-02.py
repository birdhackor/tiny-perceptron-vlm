"""核對現行導言、共用表示算例及注意力程式。"""

import contextlib
import hashlib
import io
import json
import runpy
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-03/technical-chapter-03-current-01.json"
ORIGINAL = PREVIOUS.with_name("technical-chapter-03-original-01.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(PREVIOUS) == "1b7432a86779920be3702033c970d8e91a2f8bd7771717a5b835f419cde9492f", "前次報告應保持原始版本"
assert sha(ORIGINAL) == "7a322e071cb3a292abbbb110a99aa2414ca83760d2169129d2c2c20044f1b5f6", "原始報告應保持原始版本"
previous = json.loads(PREVIOUS.read_text())
assert previous["issues"] == [], "前次報告應無待解決問題"
unchanged = {str(PREVIOUS.relative_to(ROOT)): sha(PREVIOUS), str(ORIGINAL.relative_to(ROOT)): sha(ORIGINAL)}
for artifact in previous["artifacts"]:
    path = ROOT / artifact["path"]
    assert sha(path) == artifact["sha256"], "沿用證據應與前次記錄的雜湊相同"
    unchanged[artifact["path"]] = sha(path)

chapter = "course/chapters/03.md"
base = "ecf6dec71280"
target = "c6b9339d2595"
base_text = subprocess.run(["git", "show", f"{base}:{chapter}"], capture_output=True, check=True).stdout
current_text = subprocess.run(["git", "show", f"{target}:{chapter}"], capture_output=True, check=True).stdout
assert hashlib.sha256(base_text).hexdigest() == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09", "差異起點應等於前次引用版本"
assert hashlib.sha256(current_text).hexdigest() == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5", "差異終點應等於指定現行版本"
assert current_text == (ROOT / chapter).read_bytes(), "差異終點應與現行工作樹相同"
diff_command = ["git", "diff", base, target, "--", chapter]
diff = subprocess.run(diff_command, capture_output=True, text=True, check=True).stdout

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    helpers = runpy.run_path(str(ARTIFACTS / "verify_intro.py"), run_name="__review_recheck__")
current_verification = json.loads(capture.getvalue())
old_verification = json.loads((ARTIFACTS / "verification-output.json").read_text())
assert current_verification["section_code_outputs"] == old_verification["section_code_outputs"], "注意力展示輸出應與原始核對相同"
assert helpers["blocks"](base_text.decode()) == helpers["blocks"](current_text.decode()), "章檔裡的 Python 區塊應未變動"
assert base_text.split(b"## ", 1)[0] == current_text.split(b"## ", 1)[0], "章首導言應未變動"
for heading in [
    "## 3.5 匹配特徵越多，為什麼要除以平方根？",
    "## 3.6 整段一起算，怎樣禁止讀後面的答案？",
    "## 3.7 幾種讀法怎樣並行，再交回同一位置？",
]:
    assert helpers["lesson"](base_text.decode(), heading) == helpers["lesson"](current_text.decode(), heading), "既有依據小節的程式與說明應未變動"

section = helpers["lesson"](current_text.decode(), "## 3.3 同一個位置，為什麼要有查詢與匹配兩份表示？")
namespace = {"__name__": "__review_shared_representation__"}
example_stdout = io.StringIO()
with contextlib.redirect_stdout(example_stdout):
    for block in helpers["blocks"](section):
        exec(compile(block, "查詢與匹配兩份表示算例", "exec"), namespace)
torch = helpers["torch"]
has, need = namespace["has"], namespace["need"]
shared = has @ has.T
separate = need @ has.T
assert torch.equal(shared, torch.tensor([[1.0, 0.0], [0.0, 1.0]])), "共用表示應得到自身一分、彼此零分"
assert torch.equal(separate, torch.tensor([[0.0, 1.0], [1.0, 0.0]])), "分開查詢後最高匹配應交換到另一個位置"
assert torch.equal(shared, shared.T), "同一組表示的點積分數應為對稱矩陣"
assert torch.equal(shared.argmax(dim=-1), torch.tensor([0, 1])), "本例的共用表示最高分應各自落在自己"

for source in previous["sources"]:
    if source["kind"] == "repository_code" and source["path"] != chapter:
        assert sha(ROOT / source["path"]) == source["sha256"], "其他沿用的程式來源應保持不變"
for relative, expected in unchanged.items():
    assert sha(ROOT / relative) == expected, "執行後舊報告與證據應保持不變"
attention = current_verification.pop("original_source")
current_verification.pop("command")
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-03/recheck-current-02.py",
    "executed_at": datetime.now().astimezone().isoformat(),
    "environment": current_verification["environment"],
    "previous_report_and_artifacts_unchanged": unchanged,
    "previous_issue_ids": [],
    "diff_receipt": {"command": diff_command, "base_sha256": hashlib.sha256(base_text).hexdigest(), "current_sha256": sha(ROOT / chapter), "diff": diff},
    "target_intro_unchanged": True,
    "all_chapter_python_blocks_unchanged": True,
    "relied_attention_sections_unchanged": True,
    "original_verification_rerun": current_verification,
    "shared_representation_execution": {
        "stdout": example_stdout.getvalue(),
        "has": has.tolist(),
        "need": need.tolist(),
        "shared_scores": shared.tolist(),
        "separate_scores": separate.tolist(),
        "shared_argmax_per_row": shared.argmax(dim=-1).tolist(),
        "shared_symmetric": True,
        "dtype": str(shared.dtype),
        "tolerance": "本例使用可精確表示的零與一，要求 torch.equal 精確相等。",
        "derivation": "A=(1,0)、B=(0,1)；A·A=1*1+0*0=1，B·B=0*0+1*1=1，A·B=B·A=1*0+0*1=0。各列一大於零，因此共用表示在本例的最高分皆落在自己。一般對稱性來自實數點積逐項乘法可交換；本例沒有宣稱任意向量的自身分數必定最高。"
    },
    "attention_source_retrieval": {key:value for key,value in attention.items() if key != "short_excerpts"},
    "result": "凍結導言及所有課文 Python 區塊未變；原始注意力驗證重跑通過，輸出相同；新措辭中的自身一分與彼此零分已精確驗證。"
}, ensure_ascii=False, indent=2))
