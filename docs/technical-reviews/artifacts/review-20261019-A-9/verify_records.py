"""從教材紀錄獨立核對查表答案與可辨識範圍。"""

import ast
import hashlib
import io
import itertools
import json
import platform
import re
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/A.9.md"


def section(raw, label):
    start = raw.index(f"## {label} ")
    end = raw.find("\n## ", start + 1)
    return raw[start:] if end == -1 else raw[start : end + 1]


def main():
    live = (ROOT / "course/chapters/0A.md").read_text()
    raw = FROZEN.read_text()
    assert section(live, "A.9") == raw, "凍結原文與現行小節必須相同"
    previous = section(live, "A.8")
    records = dict(re.findall(r"^\| (N|U\d+) \| (.*?) \|$", previous, re.M))
    assert len(records) == 8, "應有八筆活動室紀錄"
    clubs = {}
    floors = {}
    edges = {}
    for label, text in records.items():
        club_match = re.fullmatch(r"(.+)的活動室代碼是 ([A-Z]\d+)。", text)
        floor_match = re.fullmatch(r"([A-Z]\d+) 在(.+)。", text)
        assert bool(club_match) != bool(floor_match), "紀錄應能唯一分類"
        if club_match:
            club, code = club_match.groups()
            clubs[club] = code
            edges[club] = {"next": code, "record": label}
        else:
            code, floor = floor_match.groups()
            floors[code] = floor
            edges[code] = {"next": floor, "record": label}
    code = re.search(r"```python\n(.*?)```", raw, re.S).group(1)
    tree = ast.parse(code)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
    }
    assert assignments["code_of"] == clubs, "社團對照表應符合紀錄"
    assert assignments["floor_of"] == floors, "樓層對照表應符合紀錄"
    asked = ["攝影社", "棋藝社", "合唱社"]
    expected = [clubs[name] for name in asked]
    assert expected == ["M17", "B04", "C26"], "三筆代碼應依提問順序排列"
    paths = []
    for club in clubs:
        first = edges[club]
        second = edges[first["next"]]
        paths.append({
            "club": club,
            "code": first["next"],
            "floor": second["next"],
            "records": [first["record"], second["record"]],
        })
    assert paths[0]["floor"] == "二樓", "攝影社活動室應在二樓"
    assert paths[-1]["code"] == "E31", "美術社應對應 E31"
    assert paths[-1]["floor"] == "二樓", "美術社活動室應在二樓"
    replies = [expected, ["M17", "B04"], expected + ["E31"], ["B04", "M17", "C26"]]
    cases = []
    for reply in replies:
        cases.append({
            "reply": reply,
            "missing": [value for value in expected if value not in reply],
            "extra": [value for value in reply if value not in expected],
            "exact": reply == expected,
        })
    assert cases[0] == {"reply": expected, "missing": [], "extra": [], "exact": True}
    assert cases[1]["missing"] == ["C26"] and not cases[1]["extra"] and not cases[1]["exact"]
    assert cases[2]["extra"] == ["E31"] and not cases[2]["missing"] and not cases[2]["exact"]
    assert not cases[3]["missing"] and not cases[3]["extra"] and not cases[3]["exact"]
    permutations = list(itertools.permutations(expected))
    assert sum(list(reply) == expected for reply in permutations) == 1
    without_room_floor = {key: value for key, value in floors.items() if key != clubs["攝影社"]}
    completions = [{**without_room_floor, "M17": floor} for floor in ("一樓", "二樓", "三樓")]
    assert all(all(world[key] == value for key, value in without_room_floor.items()) for world in completions)
    assert len({world["M17"] for world in completions}) > 1, "少一筆必要紀錄時，剩餘資料不能唯一決定樓層"
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(code, str(FROZEN), "exec"), {})
    print(json.dumps({
        "environment": {"python": platform.python_version(), "device": "CPU；純 Python 字串與容器"},
        "source_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "previous_section_sha256": hashlib.sha256(previous.encode()).hexdigest(),
        "records": records,
        "code_of": clubs,
        "floor_of": floors,
        "paths": paths,
        "asked": asked,
        "expected": expected,
        "reply_cases": cases,
        "permutation_count": len(permutations),
        "exactly_matching_permutations": 1,
        "possible_answers_without_U3": [world["M17"] for world in completions],
        "same_final_answer_paths": [path for path in paths if path["floor"] == "二樓"],
        "section_stdout": output.getvalue(),
        "result": "全部核對通過；只支持人工紀錄與確定性 Python 行為，不是模型評測。",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
