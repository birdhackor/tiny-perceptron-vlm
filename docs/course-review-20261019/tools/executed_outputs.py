"""讓首讀讀者看到網站上印在每段程式正下方的實際輸出；不產生審閱判斷。

網站從執行過的 Notebook 取輸出，放在每段程式下方（`scripts/export_course.py` 的 `output_html`）；
課文原文與 reader.py 的單元裡沒有這些輸出。這支工具補上它，分兩步：

- `build`：把指定頁的程式段落對到執行過的 Notebook，寫成一份輸出清單（之後不再改寫）。
- `show`：讀者讀到某個單元後，印出已讀單元裡每段程式的輸出；還沒讀到的單元不印。
  整頁讀完、而且已用 reader.py 補讀過選讀時，另外印出折疊選讀裡的程式輸出。

驗證：清單裡每一段程式的原文，與凍結原文的 python 區塊、目前 Notebook 的程式格、執行過的
Notebook 逐段相同；`show` 只印游標已到的單元，並留下存取收據。
不涵蓋：輸出是在哪一台機器執行的（記在清單的 executed_with，網站的輸出由 CI 的 Linux 產生）；
圖片輸出只標記不顯示。
"""

import argparse
import hashlib
import importlib.util
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "docs/course-revision-20261016"
RECEIPTS = ROOT / "docs/course-review-20261019/output-access"
FENCE = re.compile(r"```python\n(.*?)```", re.S)
DETAILS = re.compile(r"<details\b[^>]*>.*?</details>", re.S)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def document(ref):
    path = ROOT / ref["path"]
    raw = path.read_bytes()
    if sha(raw) != ref["sha256"]:
        raise ValueError("檔案已改變：" + ref["path"])
    return json.loads(raw)


def frozen_page(freeze, page_id):
    page = next(p for p in freeze["pages"] if p["page_id"] == page_id)
    raw = (ROOT / page["snapshot"]).read_bytes()
    if sha(raw) != page["source_sha256"]:
        raise ValueError("凍結原文已改變：" + page_id)
    return page, raw.decode("utf-8")


def code_blocks(body):
    """本頁每個 python 區塊的程式原文（去頭尾空白，和 build_course 寫進 Notebook 的一樣）與是否在折疊選讀裡。"""
    folded = [match.span() for match in DETAILS.finditer(body)]
    return [
        {"code": match[1].strip(), "in_details": any(a <= match.start() < b for a, b in folded)}
        for match in FENCE.finditer(body)
    ]


def program_cells(notebook):
    """Notebook 裡對應課文 python 區塊的程式格：跳過環境準備格與圖解格。"""
    return [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and not cell.get("metadata", {}).get("course_setup")
        and "course_figure" not in cell.get("metadata", {})
    ]


def outputs_of(cell):
    result = []
    for output in cell.get("outputs", []):
        kind = output["output_type"]
        if kind == "stream":
            result.append({"kind": output["name"], "text": "".join(output["text"])})
        elif kind in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "image/png" in data or "image/svg+xml" in data:
                result.append({"kind": "image", "text": "（網站在這裡顯示一張圖；這個工具不顯示圖）"})
            elif "text/plain" in data:
                result.append({"kind": "result", "text": "".join(data["text/plain"])})
        elif kind == "error":
            raise ValueError("執行過的 Notebook 有錯誤輸出")
    return result


def build(args):
    freeze_ref = {"path": args.freeze.as_posix(), "sha256": sha((ROOT / args.freeze).read_bytes())}
    freeze = document(freeze_ref)
    index = {item["id"]: item for item in json.loads((ROOT / "course/lesson-index.json").read_bytes())}
    pages = {}
    for page_id in args.pages:
        page, body = frozen_page(freeze, page_id)
        blocks = code_blocks(body)
        entry = {"source_sha256": page["source_sha256"], "notebook": None, "blocks": []}
        if page_id in index:
            relative = Path(index[page_id]["notebook"])
            current_raw = (ROOT / relative).read_bytes()
            executed_path = args.executed / relative.relative_to("notebooks")
            executed_raw = executed_path.read_bytes()
            current, executed = json.loads(current_raw), json.loads(executed_raw)
            if [c["source"] for c in current["cells"]] != [c["source"] for c in executed["cells"]]:
                raise ValueError("執行過的 Notebook 不是目前的版本：" + page_id)
            cells = program_cells(executed)
            if [block["code"] for block in blocks] != ["".join(c["source"]).strip() for c in cells]:
                raise ValueError("課文的 python 區塊與 Notebook 程式格對不上：" + page_id)
            if any(c.get("execution_count") is None for c in cells):
                raise ValueError("Notebook 有程式格沒有執行：" + page_id)
            entry["notebook"] = {
                "path": relative.as_posix(),
                "sha256": sha(current_raw),
                "executed_sha256": sha(executed_raw),
            }
            entry["blocks"] = [
                {
                    "index": number,
                    "in_details": block["in_details"],
                    "code_sha256": sha(block["code"].encode()),
                    "outputs": outputs_of(cell),
                }
                for number, (block, cell) in enumerate(zip(blocks, cells))
            ]
        elif blocks:
            raise ValueError("有程式卻沒有 Notebook：" + page_id)
        pages[page_id] = entry
    bundle = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "首讀讀者在網站上會看到的程式輸出；由 executed_outputs.py show 依讀者游標逐段顯示。",
        "source_freeze": freeze_ref,
        "executed_with": args.executed_with,
        "pages": pages,
    }
    with (ROOT / args.output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pages": len(pages), "blocks": sum(len(p["blocks"]) for p in pages.values())}))


def reader_units(body):
    """與 reader.py 相同的單元切法（直接載入它的 units）。"""
    spec = importlib.util.spec_from_file_location("reader_tool", BATCH / "tools/reader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.units(body)


def supplement_read(state):
    for path in (BATCH / "supplement-access").glob("*.json"):
        receipt = json.loads(path.read_bytes())
        if (
            receipt.get("reviewer_task") == state["reviewer_task"]
            and receipt.get("reading_plan") == state["reading_plan"]
            and receipt.get("page_id") == state["page_id"]
        ):
            return True
    return False


def show(args):
    if not re.fullmatch(r"[a-f0-9]{32}", args.session):
        raise ValueError("無效 session")
    state = json.loads((BATCH / "reader-sessions" / args.session / "state.json").read_bytes())
    if state.get("reviewer_task") != args.reviewer:
        raise ValueError("只能看自己的 session")
    plan = document(state["reading_plan"])
    if "executed_outputs" not in plan:
        raise ValueError("這份閱讀計畫沒有指定程式輸出清單")
    bundle = document(plan["executed_outputs"])
    if bundle["source_freeze"] != state["source_freeze"]:
        raise ValueError("輸出清單與 session 的凍結版本不同")
    freeze = document(state["source_freeze"])
    page, body = frozen_page(freeze, state["page_id"])
    entry = bundle["pages"][state["page_id"]]
    if page["source_sha256"] != state["source_sha256"] or entry["source_sha256"] != state["source_sha256"]:
        raise ValueError("輸出清單不是這一版原文")
    chunks = reader_units(body)
    reached = min(state["next_index"] + 1, len(chunks))
    main_blocks = [block for block in entry["blocks"] if not block["in_details"]]
    printed, cursor = [], 0
    for unit_index, chunk in enumerate(chunks[:reached]):
        for code in FENCE.findall(chunk):
            block = main_blocks[cursor]
            cursor += 1
            if sha(code.strip().encode()) != block["code_sha256"]:
                raise ValueError("單元裡的程式與輸出清單對不上")
            printed.append(block["index"])
            print(f"【單元 {unit_index} 的第 {block['index'] + 1} 段程式：網站上印在它正下方的實際執行結果 · CPU】")
            print("".join(item["text"] for item in block["outputs"]) or "（這段程式沒有印出任何東西）")
    if state["next_index"] >= len(chunks) and supplement_read(state):
        for block in entry["blocks"]:
            if block["in_details"]:
                printed.append(block["index"])
                print(f"【折疊選讀裡的第 {block['index'] + 1} 段程式：實際執行結果 · CPU】")
                print("".join(item["text"] for item in block["outputs"]) or "（這段程式沒有印出任何東西）")
    if not printed:
        print("（到目前為止讀到的單元裡沒有程式）")
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    with (RECEIPTS / (uuid.uuid4().hex + ".json")).open("x", encoding="utf-8") as stream:
        receipt = {
            "reviewer_task": args.reviewer,
            "session": args.session,
            "page_id": state["page_id"],
            "units_reached": reached,
            "blocks_printed": printed,
            "executed_outputs": plan["executed_outputs"],
            "accessed_at": datetime.now(UTC).isoformat(),
        }
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    make = sub.add_parser("build")
    make.add_argument("--freeze", type=Path, required=True, help="repo 相對路徑")
    make.add_argument("--pages", nargs="+", required=True)
    make.add_argument("--executed", type=Path, required=True, help="check_notebooks.py --output 的目錄")
    make.add_argument("--executed-with", required=True, help="執行指令與環境（Python、torch、平台）")
    make.add_argument("--output", type=Path, required=True, help="repo 相對路徑，不覆寫")
    view = sub.add_parser("show")
    view.add_argument("session")
    view.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    build(args) if args.operation == "build" else show(args)


if __name__ == "__main__":
    main()
