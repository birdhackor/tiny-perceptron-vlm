"""凍結閱讀範圍、核對獨立 AI 估時，並在網站呈現來源綁定的時間區間。

本程式不估算時間。估時者必須閱讀 inventory 的正文快照與圖解，再填寫外部
JSON；不能把字數、程式行數或自動導覽當成已完成的閱讀評審。

估時 JSON 的頂層是 schema_version=1、estimate_kind="ai_estimate"、pages 陣列。
每筆 pages 紀錄必填 page_id、minutes_min、minutes_max、reviewer_task、reason、
source_sha256、figures_sha256。分鐘是正整數；圖檔 map 以 repo 相對路徑為 key，
沒有圖時也明寫空 map。reviewer_task 與 reason 記錄誰讀了什麼、如何判斷區間。
所有 SHA 必須與目前 inventory 相同；不能由本程式補出未經閱讀的分鐘數。

選用的路線 JSON 是 schema_version=1、routes 陣列；每條路線必填 id、label、
page_ids。page_ids 只使用 inventory 的公開頁 ID，不使用章號範圍或導航標題。
inventory --executed 對應 exporter --executed 的首頁版本。
"""

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "course/reading-times.json"
ROUTES = ROOT / "course/reading-time-routes.json"
SCHEMA_VERSION = 1
COURSE_DOCUMENTS = {
    "course",
    "first-steps",
    "training",
    "glossary",
    "natural-v4-student",
    "natural-v4-data",
    "natural-v4-training",
}
METHOD = (
    "這是 AI 對閱讀時間的估計，不是實測平均。以數學基礎良好的高中生，或具基本數學背景的大學生為對象，"
    "包含讀正文、看圖、理解範例程式與輸出，以及短暫停下來思考。"
    "不包含安裝、下載、親手修改或執行練習、等待訓練，或額外補讀前置教材的時間；"
    "實際速度會隨背景與熟悉程度改變。總量把列出的教材頁面各算一次，章導讀只算獨立前言，"
    "共用導覽與重複索引不重算。路線總量只含該路線明列的頁面，重複頁面只算一次。"
)


def source_sha256(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chapter_introduction(content):
    return re.split(r"^## ", content, maxsplit=1, flags=re.M)[0]


def lesson_slices(content):
    """保留小節標題與原始空白，與讀者評審使用同一段原始正文。"""
    headings = list(re.finditer(r"^## ([A-Z\d]+\.\d+) .+$", content, re.M))
    result = {}
    for position, heading in enumerate(headings):
        if heading[1] in result:
            raise ValueError(f"重複小節 ID：{heading[1]}")
        end = headings[position + 1].start() if position + 1 < len(headings) else len(content)
        result[heading[1]] = content[heading.start() : end]
    return result


def figure_hashes(root, source, content):
    """只核對這段正文實際引用的本機圖解，不把整章圖集套到每一頁。"""
    hashes = {}
    references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
    references += re.findall(r'<img\b[^>]*\bsrc=[\'"]([^\'"]+)[\'"]', content)
    for reference in references:
        if "://" in reference or reference.startswith("data:"):
            continue
        path = (source.parent / reference.split("#", 1)[0]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"閱讀來源引用不存在的本機圖檔：{source}: {reference}")
        hashes[path.relative_to(root.resolve()).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def build_inventory(root, index, documents, home_content, snapshots=None):
    """每個公開 URL 只有一個 canonical ID；快照沒有估時或生成導覽。"""
    root = root.resolve()
    pages = []
    texts = {}
    sources = {item["source"]: (root / item["source"]).read_text(encoding="utf-8") for item in index}
    sections = {source: lesson_slices(text) for source, text in sources.items()}

    def add(page_id, kind, source, selector, content, title):
        if page_id in texts:
            raise ValueError(f"重複 canonical page ID：{page_id}")
        path = root / source
        pages.append(
            {
                "page_id": page_id,
                "page_path": page_id + ".md",
                "kind": kind,
                "title": title,
                "source": source,
                "selector": selector,
                "source_sha256": source_sha256(content),
                "figures_sha256": figure_hashes(root, path, content),
            }
        )
        texts[page_id] = content

    for item in index:
        try:
            content = sections[item["source"]][item["id"]]
        except KeyError as error:
            raise ValueError(f"lesson-index 沒有對應的原始正文：{item['id']}") from error
        add(item["id"], "lesson", item["source"], "section:" + item["id"], content, item["title"])
    for source, text in sources.items():
        intro = chapter_introduction(text)
        add(
            "chapter-" + Path(source).stem,
            "chapter_introduction",
            source,
            "before-first-section",
            intro,
            intro.splitlines()[0].removeprefix("# "),
        )
    for page_id, source in documents.items():
        text = (root / source).read_text(encoding="utf-8")
        add(
            page_id,
            "course_guide" if page_id in COURSE_DOCUMENTS else "reference",
            source,
            "whole-file",
            text,
            text.splitlines()[0].removeprefix("# "),
        )
    add(
        "index",
        "home",
        "scripts/export_course.py",
        "home-introduction-without-chapter-navigation",
        home_content,
        home_content.splitlines()[0].removeprefix("# "),
    )
    if snapshots is not None:
        snapshots = snapshots.resolve()
        snapshots.mkdir(parents=True, exist_ok=True)
        for page in pages:
            path = snapshots / page["page_path"]
            path.write_text(texts[page["page_id"]], encoding="utf-8")
            page["snapshot"] = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    course_ids = [
        page["page_id"] for page in pages if page["kind"] in {"lesson", "chapter_introduction", "course_guide"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_hash_scheme": "sha256-of-utf8-authored-markdown-slice",
        "counts": dict(sorted(Counter(page["kind"] for page in pages).items())),
        "pages": pages,
        "groups": [
            {"id": "canonical_course", "label": "全部教材正文", "page_ids": course_ids},
            {"id": "complete_site", "label": "全部網站頁面（含首頁與技術參考）", "page_ids": list(texts)},
        ],
    }


def load_estimates(path, inventory, require_complete=False):
    """缺少估時可以建置；已提供的估時必須完整有效，不能悄悄使用過期資料。"""
    expected = {page["page_id"]: page for page in inventory["pages"]}
    if not path.is_file():
        if require_complete:
            raise ValueError(f"缺少閱讀時間 metadata：{path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != SCHEMA_VERSION
        or data.get("estimate_kind") != "ai_estimate"
    ):
        raise ValueError("閱讀時間 metadata 必須使用 schema_version=1 與 estimate_kind=ai_estimate")
    if not isinstance(data.get("pages"), list):
        raise ValueError("閱讀時間 metadata 的 pages 必須是陣列")
    result = {}
    for record in data["pages"]:
        if not isinstance(record, dict):
            raise ValueError("閱讀時間每筆紀錄必須是 JSON object")
        page_id = record.get("page_id")
        if not isinstance(page_id, str) or page_id not in expected or page_id in result:
            raise ValueError(f"未知或重複的閱讀時間 page_id：{page_id}")
        lower, upper = record.get("minutes_min"), record.get("minutes_max")
        if type(lower) is not int or type(upper) is not int or lower < 1 or upper < lower:
            raise ValueError(f"{page_id}: minutes_min/max 必須是正整數，且上限不得小於下限")
        for field in ("reviewer_task", "reason"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{page_id}: 缺少 {field}")
        if record.get("source_sha256") != expected[page_id]["source_sha256"]:
            raise ValueError(f"{page_id}: 閱讀時間評審不是目前正文")
        if record.get("figures_sha256") != expected[page_id]["figures_sha256"]:
            raise ValueError(f"{page_id}: 閱讀時間評審的圖檔 SHA 不完整或已過期")
        result[page_id] = record
    missing = set(expected) - set(result)
    if require_complete and missing:
        raise ValueError(f"缺少 {len(missing)} 頁閱讀時間：{', '.join(sorted(missing))}")
    return result


def load_routes(path, inventory):
    """路線以實際公開頁面 ID 定義，重複 ID 在合計前去重。"""
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != SCHEMA_VERSION
        or not isinstance(data.get("routes"), list)
    ):
        raise ValueError("閱讀路線必須使用 schema_version=1 與 routes 陣列")
    pages = {page["page_id"] for page in inventory["pages"]}
    seen = set()
    routes = []
    for route in data["routes"]:
        if not isinstance(route, dict):
            raise ValueError("每條閱讀路線必須是 JSON object")
        route_id = route.get("id")
        if not isinstance(route_id, str) or not route_id.strip() or route_id in seen:
            raise ValueError(f"未知或重複的閱讀路線 id：{route_id}")
        if not isinstance(route.get("label"), str) or not route["label"].strip():
            raise ValueError(f"{route_id}: 路線缺少 label")
        ids = route.get("page_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in pages for i in ids):
            raise ValueError(f"{route_id}: page_ids 必須列出存在的 canonical page ID")
        routes.append({"id": route_id, "label": route["label"], "page_ids": list(dict.fromkeys(ids))})
        seen.add(route_id)
    return routes


def total_range(page_ids, estimates):
    ids = list(dict.fromkeys(page_ids))
    if not ids or any(page_id not in estimates for page_id in ids):
        return None
    return (sum(estimates[i]["minutes_min"] for i in ids), sum(estimates[i]["minutes_max"] for i in ids))


def format_range(minutes, *, total=False):
    lower, upper = minutes
    if total and upper >= 60:
        # 一位小數足以規劃課程；使用整數運算向外取整，不縮小估計區間。
        lower_tenths, upper_tenths = lower // 6, (upper + 5) // 6
        return f"{lower_tenths / 10:.1f}–{upper_tenths / 10:.1f} 小時"
    return f"{lower}–{upper} 分鐘"


def render_page(content, page_id, estimates, inventory, routes=()):
    """只在生成網站插入呈現；原始正文、Notebook 和評審紀錄保持分離。"""
    pieces = []
    if page_id in estimates:
        estimate = estimates[page_id]
        pieces.append(
            '<p class="reading-time"><small>閱讀約 '
            + format_range((estimate["minutes_min"], estimate["minutes_max"]))
            + " · AI 估計</small></p>"
        )
    if page_id in {"index", "course"}:
        rows = []
        for group in [*inventory["groups"], *routes]:
            minutes = total_range(group["page_ids"], estimates)
            if minutes is not None:
                label = html.escape(group["label"])
                rows.append(f"<li>{label}：約 {format_range(minutes, total=True)}</li>")
        if rows:
            course_group = next(group for group in inventory["groups"] if group["id"] == "canonical_course")
            course_minutes = total_range(course_group["page_ids"], estimates)
            summary = "閱讀路線與時間"
            if course_minutes is not None:
                summary = "全教材約 " + format_range(course_minutes, total=True) + " · AI 估計"
            pieces.append(
                '<details class="reading-time-method"><summary>'
                + summary
                + '</summary><div class="reading-time-totals"><p>閱讀總量 · AI 估計</p><ul>'
                + "".join(rows)
                + "</ul></div><p>"
                + METHOD
                + "</p></details>"
            )
        else:
            pieces.append(
                '<details class="reading-time-method"><summary>閱讀時間怎麼估？</summary><p>'
                + METHOD
                + "</p></details>"
            )
    if not pieces:
        return content
    banner = "\n\n" + "\n\n".join(pieces) + "\n\n"
    heading = re.search(r"^# [^\n]+\n?", content, re.M)
    if heading:
        return content[: heading.end()] + banner + content[heading.end() :]
    return banner.lstrip() + content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory", help="輸出無估時的凍結頁清單與可逐頁閱讀的原始正文快照")
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.add_argument("--snapshots", type=Path, required=True)
    inventory_parser.add_argument("--executed", action="store_true", help="首頁使用已核對 CPU 執行結果的建置說明")
    validate_parser = subparsers.add_parser("validate", help="要求每個發布頁都有目前來源與圖檔綁定的估時")
    validate_parser.add_argument("--metadata", type=Path, default=METADATA)
    validate_parser.add_argument("--routes", type=Path, default=ROUTES)
    validate_parser.add_argument("--executed", action="store_true")
    args = parser.parse_args()
    # 延後引入，讓 exporter 可以使用這個模組而不形成 import cycle。
    if __package__:
        from scripts.export_course import DOCUMENTS, home_introduction
    else:
        from export_course import DOCUMENTS, home_introduction

    index = json.loads((ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    snapshots = args.snapshots if args.command == "inventory" else None
    inventory = build_inventory(ROOT, index, DOCUMENTS, home_introduction(index, args.executed), snapshots)
    if args.command == "inventory":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"凍結 {len(inventory['pages'])} 個發布頁至 {args.output}；正文快照：{args.snapshots}；未估算閱讀時間")
    else:
        estimates = load_estimates(args.metadata, inventory, require_complete=True)
        routes = load_routes(args.routes, inventory)
        print(f"閱讀時間核對通過：{len(estimates)} 頁，{len(routes)} 條路線；來源與圖檔 SHA 均為目前版本")


if __name__ == "__main__":
    main()
