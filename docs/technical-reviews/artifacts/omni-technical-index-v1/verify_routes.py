"""核首頁宣告的教學路線與兩個必要前提的當版程式；不重審全書。"""

import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def section(path, heading):
    raw = (ROOT / path).read_text()
    start = re.search(r"^## " + re.escape(heading) + r" .+$", raw, re.M).start()
    following = re.search(r"^## ", raw[start + 3:], re.M)
    return raw[start:start + 3 + following.start()] if following else raw[start:]


def first_code(path, heading):
    code = re.search(r"```python\n(.*?)\n```", section(path, heading), re.S)[1]
    output = io.StringIO()
    names = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, path + "#" + heading + ":first-python-fence", "exec"), names)
    return names, {"source": path + "#" + heading, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                   "stdout": output.getvalue()}


def main():
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    ids = [x["id"] for x in index]
    assert ids.index("2.2") < ids.index("7.1") < ids.index("7.11")
    guide = (ROOT / "course/README.md").read_text()
    route_checks = {
        "context_before_chat": ["2.2", "7.1", "7.11"],
        "information_sufficiency": ["9.13"],
        "image_audio_time": ["10.1", "12.1", "12.17"],
        "long_material_and_memory": ["16.2", "16.3"],
        "persona_safety_quantization": ["8.1", "9.1", "17.1"],
    }
    for group in route_checks.values():
        assert all(x in ids for x in group)
    for phrase in ["如何教文風、誠實、安全與偏好", "圖片與聲音怎麼成為輸入", "第 17 章", "CPU機制參考", "正式新模型訓練與成熟模型實測仍待完成"]:
        assert phrase in guide
    chat, chat_receipt = first_code("course/chapters/07.md", "7.1")
    assert chat["x"].tolist() == [1, 3, 57, 51, 57, 69, 71, 2, 4, 58]
    assert chat["y"][chat["y"] != chat["IGNORE"]].tolist() == [58, 2]
    _, sufficiency_receipt = first_code("course/chapters/09.md", "9.13")
    text = sufficiency_receipt["stdout"]
    assert "資訊充分 no 下一步 詢問申請編號" in text
    assert "資訊充分 不適用 下一步 轉給真人客服" in text
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_routes.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu"},
        "route_ids_present": route_checks, "route_order": "2.2 before 7.1 before 7.11 in current lesson index",
        "guide_persona_safety_image_quantization_routes_present": True,
        "guide_new_formal_training_boundary_present": True,
        "chat": chat_receipt, "rule_specific_sufficiency": sufficiency_receipt,
        "scope": "核首頁路線承諾及當版7.1/9.13的首段機制；沒有核所有先決節或一般模型能力，沒有訓練。",
    }
    Path(__file__).with_name("route-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
