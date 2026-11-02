"""20.15 技術審查：固定來源、教材原碼、儲存練習與圖像收據；不載入模型。"""

import contextlib
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
CACHE = Path("/workspace/work/reference/omni-technical-20-15-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def current_slice(source, lesson_id):
    raw = source.read_bytes()
    match = re.search(rb"^## " + re.escape(lesson_id.encode()) + rb" .*$", raw, re.M)
    assert match is not None
    end = re.search(rb"^## ", raw[match.end():], re.M)
    end_at = match.end() + end.start() if end else len(raw)
    return raw[match.start():end_at]


freeze = json.loads(FREEZE.read_bytes())
pages = {p["page_id"]: p for p in freeze["pages"]}
scope = {
    "reviewer_task": "/root/omni_technical_coordinator/technical_20_15",
    "created_at": datetime.now(UTC).isoformat(),
    "source_freeze": str(FREEZE.relative_to(ROOT)),
    "source_freeze_sha256": digest(FREEZE),
    "reader_task": "/root/omni_reader_mature",
    "pages": [],
    "scope_note": "20.15 完整審查；14.19/17.17/7.20 僅核明連必要語義，19.21/19.22 僅核第19章分工的上下文規格，沒有替那些頁作完整技術審查。",
}
semantic_quotes = {
    "14.19": "把「座標」從一個序號擴成三種：時間 `t`、圖塊所在的高度 `h`、寬度 `w`。",
    "17.17": "此處 MTP 的工作是補同框細節。",
    "7.20": "在同一個前文位置，要求模型預測不止一個未來的 token。",
    "19.21": "這些是計算檢查，不是解題成績。",
    "19.22": "Talker依條件預測聲音碼，凍結的 decoder 再將碼還原成8kHz波形。",
}
for page_id in ["20.15", "14.19", "17.17", "7.20", "19.21", "19.22"]:
    page = pages[page_id]
    snapshot = ROOT / page["snapshot"]
    body = snapshot.read_bytes()
    actual = current_slice(ROOT / page["source"], page_id)
    assert digest(snapshot) == page["source_sha256"]
    assert actual == body, page_id
    quote = semantic_quotes.get(page_id)
    if quote:
        assert quote in body.decode()
    scope["pages"].append({
        "page_id": page_id,
        "source": page["source"],
        "snapshot": page["snapshot"],
        "source_sha256": page["source_sha256"],
        "current_slice_sha256": hashlib.sha256(actual).hexdigest(),
        "current_slice_equals_snapshot": True,
        "semantic_quote": quote,
    })

page = pages["20.15"]
body = (ROOT / page["snapshot"]).read_text()
code_blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code_blocks) == 1
code = code_blocks[0]
code_path = OUT / "inline-code.py"
assert not code_path.exists()
code_path.write_text(code)
notebook = ROOT / page["notebook"]
assert digest(notebook) == page["notebook_sha256"]
cells = json.loads(notebook.read_bytes())["cells"]
nb_codes = ["".join(c["source"]).strip() for c in cells if c["cell_type"] == "code"]
assert code.strip() in nb_codes
scope["notebook"] = {
    "path": page["notebook"],
    "sha256": digest(notebook),
    "inline_code_exact_match_after_outer_whitespace": True,
}
scope["inline_code"] = {
    "path": str(code_path.relative_to(ROOT)),
    "sha256": digest(code_path),
    "source_lines": "20.15 snapshot L20-L27",
    "extraction": "Only Python fenced block, preserving its bytes between fence lines.",
}
save("scope-receipt.json", scope)

environment = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "device": "CPU; integer arithmetic only; no model loaded",
    "torch_installed_but_not_used": importlib.metadata.version("torch"),
}
baseline_command = [sys.executable, str(code_path.relative_to(ROOT))]
baseline = subprocess.run(baseline_command, cwd=ROOT, text=True, capture_output=True, check=True)
expected = "全部權重名義值GB 60.0\n全部權重名義值GiB 55.88\n只按啟用量算GB 6.0\n"
assert baseline.stdout == expected
assert not baseline.stderr
assert code.count("bytes_per_parameter = 2") == 1
exercise_code = code.replace("bytes_per_parameter = 2", "bytes_per_parameter = 4")
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code, "20.15-exercise-bytes-4", "exec"), {})
exercise_expected = "全部權重名義值GB 120.0\n全部權重名義值GiB 111.76\n只按啟用量算GB 12.0\n"
assert exercise_stdout.getvalue() == exercise_expected
save("inline-execution.json", {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-15-v1/verify_material.py",
    "environment": environment,
    "code_path": str(code_path.relative_to(ROOT)),
    "code_sha256": digest(code_path),
    "inputs": {"total_parameters": 30_000_000_000, "active_parameters": 3_000_000_000, "bytes_per_parameter": 2},
    "baseline_subprocess": {"argv": baseline_command, "returncode": baseline.returncode, "stdout": baseline.stdout, "stderr": baseline.stderr},
    "exercise": {"change": "Only bytes_per_parameter = 2 becomes 4; compiled same inline block", "stdout": exercise_stdout.getvalue()},
    "result": "Both exact expected stdout comparisons passed. No helpers/models/trainers used.",
    "derivation": {
        "base_bytes": 30_000_000_000 * 2,
        "base_GB": 60.0,
        "base_GiB_unrounded": 60_000_000_000 / 1024**3,
        "active_only_GB": 6.0,
        "exercise_GB_GiB_activeGB": [120.0, 111.76, 12.0],
        "scope": "Nominal weight storage under lesson assumptions; no measured GPU memory, speed, quality or exact Qwen parameter census.",
    },
})

figure = "course/figures/omni-20-architecture.svg"
frozen_figure = freeze["figure_snapshots"][figure] if "figure_snapshots" in freeze else None
if frozen_figure is None:
    # The freeze uses a map whose key is the original figure path.
    maps = [v for v in freeze.values() if isinstance(v, dict) and figure in v]
    assert len(maps) == 1
    frozen_figure = maps[0][figure]
png = CACHE / "omni-20-architecture.png"
assert digest(ROOT / figure) == page["figures_sha256"][figure]
assert digest(ROOT / frozen_figure["path"]) == page["figures_sha256"][figure]
assert png.exists()
save("figure-render-receipt.json", {
    "created_at": datetime.now(UTC).isoformat(),
    "source": figure,
    "source_sha256": digest(ROOT / figure),
    "frozen_source": frozen_figure,
    "command": "inkscape docs/course-revision-20261016/frozen-figures/75586e0713ce9a4dd8095d25b444cea3d3f8dd04db4e26db8d6675f828361861.svg --export-type=png --export-filename=/workspace/work/reference/omni-technical-20-15-v1/omni-20-architecture.png --export-width=760",
    "renderer": subprocess.run(["inkscape", "--version"], text=True, capture_output=True, check=True).stdout.strip(),
    "render_exit_code": 0,
    "render_stderr": "Non-fatal PangoFT2FontMap/GtkRecentManager wrapper warnings; PNG rendered successfully.",
    "png_path_outside_git": str(png),
    "png_sha256": digest(png),
    "viewed_with": "tools.view_image via functions.exec, 2026-10-29; actual image output inspected",
    "inspection": "Both vertical panels show downward data transfer, not speed or scale. 2.5 dual text conditions, Talker codes, DiT mel with 1 future block, BigVGAN waveform; 3 text plus audio/visual conditions, zeroth then same-frame residual codebooks, causal Code2Wav waveform. Labels fit boxes and arrows connect intended stages; no comparative timing/parameter scale encoded.",
})
print(json.dumps({"scope_checks": "passed", "inline_baseline": baseline.stdout.splitlines(), "exercise_bytes4": exercise_stdout.getvalue().splitlines(), "render_sha256": digest(png)}, ensure_ascii=False))
