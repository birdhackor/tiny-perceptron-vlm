"""重讀修訂句的原始依據，核對凍結差異，並重跑既有離線算例。"""

import contextlib
import concurrent.futures
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).parent
ROOT = Path.cwd()
REPORT = Path("docs/course-review-20261019/reviews/rv1019_technical_glossary/technical-glossary-original-01.json")
OLD = Path("docs/course-review-20261019/freeze-01/sources/glossary.md")
NEW = Path("docs/course-review-20261019/freeze-05/sources/glossary.md")
FREEZE = Path("docs/course-review-20261019/freeze-05.json")
COMMAND = "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-glossary/recheck-glossary-current-01.py"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

for name in ["verification-output-current-01.json", "sources-current-01.json", "diff-current-01.patch", "audit-output-current-01.json"]:
    if (BASE / name).exists():
        raise FileExistsError("輸出已存在，請使用新的檔名：" + name)

report = json.loads(REPORT.read_text())
assert sha(REPORT) == "114292e801fcc71de70bfae0874b3cc70192f7668c7917f4f8250e026ece914e"
assert sha(OLD) == "77ef581c3452cbaf1d8b06abf3efe18a87670ce48bcda3e7eb01242f07f3bd1d"
assert sha(NEW) == "6985f5f3137fe8fa3937b3e03d8cafb985e6f8787a4e6db675199a650514e037"
assert sha(FREEZE) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
assert Path("course/glossary.md").read_bytes() == NEW.read_bytes()

audit = {
    "command": COMMAND,
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
    "original_report": {"path": str(REPORT), "sha256": sha(REPORT)},
    "source": {"path": str(NEW), "sha256": sha(NEW)},
    "source_freeze": {"path": str(FREEZE), "sha256": sha(FREEZE)},
    "repo_source_hashes": [],
    "inherited_artifact_hashes": [],
}
for source in report["sources"]:
    if source["kind"] == "repository_code":
        current = sha(source["path"])
        audit["repo_source_hashes"].append({"path": source["path"], "expected": source["sha256"], "observed": current})
        assert current == source["sha256"], source["path"]
for artifact in report["artifacts"]:
    current = sha(artifact["path"])
    audit["inherited_artifact_hashes"].append({"path": artifact["path"], "expected": artifact["sha256"], "observed": current})
    assert current == artifact["sha256"], artifact["path"]

diff_command = ["git", "diff", "--no-index", str(OLD), str(NEW)]
diff = subprocess.run(diff_command, capture_output=True)
assert diff.returncode == 1 and not diff.stderr
with (BASE / "diff-current-01.patch").open("xb") as stream:
    stream.write(diff.stdout)
old_lines, new_lines = OLD.read_text().splitlines(), NEW.read_text().splitlines()
assert len(old_lines) == len(new_lines)
changed = [{"before": before, "after": after} for before, after in zip(old_lines, new_lines) if before != after]
assert len(changed) == 2
audit["diff"] = {"command": " ".join(diff_command), "exit_code": diff.returncode, "path": str(BASE / "diff-current-01.patch"), "changed_rows": changed}
audit["figures"] = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", NEW.read_text())
assert not audit["figures"]

class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

sources = [
    ("unicode_current", "https://www.unicode.org/reports/tr29/tr29-47.html", ["A single Unicode code point is often", "There are many cases", "GB9"]),
    ("utf8_current", "https://www.rfc-editor.org/rfc/rfc3629.txt", ["The definition of UTF-8 prohibits", "0000 0800-0000 FFFF", "0001 0000-0010 FFFF"]),
    ("ppo_current", "https://spinningup.openai.com/en/latest/algorithms/ppo.html", ["doesn’t have a constraint at all", "still possible", "early stopping"]),
    ("gcb_current", "https://www.unicode.org/Public/17.0.0/ucd/auxiliary/GraphemeBreakProperty.txt", ["GraphemeBreakProperty-17.0.0", "1F3FB..1F3FF"]),
]
def fetch(source):
    name, url, needles = source
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    response = subprocess.run(command, capture_output=True)
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))
    raw = response.stdout
    plain = raw.decode("utf-8")
    if "<html" in plain[:1000].lower() or "<!doctype" in plain[:1000].lower():
        parser = Plain()
        parser.feed(plain)
        plain = " ".join(parser.parts)
    plain = re.sub(r"\s+", " ", plain)
    windows = []
    for needle in needles:
        found = re.search(re.escape(needle), plain, re.I)
        if found:
            windows.append({"locator": needle, "text": plain[max(0, found.start()-60):found.end()+850]})
    assert windows, name
    receipt = {"id": name, "url": url, "command": " ".join(command), "accessed_at": datetime.now(timezone.utc).isoformat(), "exit_code": response.returncode, "retrieved_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "located_terms": [w["locator"] for w in windows], "short_quote": " ".join(windows[0]["text"].split()[:24])}
    if name == "gcb_current":
        line = next(line for line in raw.decode().splitlines() if line.startswith("1F3FB..1F3FF"))
        assert "; Extend " in line or "; Extend #" in line
        receipt["modifier_property"] = line
    return receipt, windows

records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for receipt, windows in pool.map(fetch, sources):
        records.append(receipt)
        print(json.dumps({"id": receipt["id"], "reading": windows}, ensure_ascii=False))
save_json(BASE / "sources-current-01.json", records)
audit["sources"] = [{"id": r["id"], "url": r["url"], "retrieved_sha256": r["retrieved_sha256"]} for r in records]

text = "👍🏽"
audit["emoji"] = {"text": text, "codepoints": [f"U+{ord(c):04X}" for c in text], "python_length": len(text), "utf8_bytes": list(text.encode("utf-8")), "utf8_byte_count": len(text.encode("utf-8")), "grapheme_reason": "膚色碼點U+1F3FD屬Extend，UAX #29的GB9在它前面不設grapheme boundary；兩碼點可構成同一extended grapheme cluster。"}
assert audit["emoji"]["codepoints"] == ["U+1F44D", "U+1F3FD"] and len(text.encode()) == 8
length_counts = {n: 0 for n in range(1, 5)}
surrogate_errors = 0
for value in range(0x110000):
    if 0xD800 <= value <= 0xDFFF:
        try:
            chr(value).encode("utf-8")
        except UnicodeEncodeError:
            surrogate_errors += 1
        else:
            raise AssertionError("surrogate不應可直接編碼")
    else:
        encoded = chr(value).encode("utf-8")
        assert 1 <= len(encoded) <= 4
        assert encoded.decode("utf-8") == chr(value)
        length_counts[len(encoded)] += 1
assert surrogate_errors == 2048
audit["utf8_exhaustive"] = {"range": "U+0000..U+10FFFF", "valid_scalar_count": sum(length_counts.values()), "byte_length_counts": length_counts, "surrogate_range": "U+D800..U+DFFF", "surrogate_errors": surrogate_errors, "result": "所有有效scalar的UTF-8均為1–4bytes且可還原；全部surrogate均拒絕直接編碼。"}

original_program = BASE / "verify_glossary.py"
code = original_program.read_text()
changes = [
    ('docs/course-review-20261019/freeze-01/sources/glossary.md', str(NEW)),
    ('PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-glossary/verify_glossary.py', COMMAND),
    ('(BASE/"verification-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\\n")', 'with (BASE/"verification-output-current-01.json").open("x") as stream:\n    stream.write(json.dumps(out, ensure_ascii=False, indent=2)+"\\n")'),
    ('所有離線斷言通過；PPO 與 Unicode 的反例保留為技術問題的證據。', '所有離線斷言通過；多碼點與PPO clipping行為支持新版限制說明。'),
]
for before, after in changes:
    assert before in code, before
    code = code.replace(before, after)
audit["replayed_program"] = {"path": str(original_program), "sha256": sha(original_program), "in_memory_substitutions": [{"before": x, "after": y} for x, y in changes], "note": "只在記憶體換本次凍結輸入、命令與新輸出位置，原程式與原輸出完全保留。"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, str(original_program), "exec"), {"__name__": "__main__", "__file__": str(original_program)})
replay = json.loads((BASE / "verification-output-current-01.json").read_text())
audit["environment"]["torch"] = replay["environment"]["torch"]
assert replay["freeze"]["sha256"] == sha(NEW)
audit["replayed_result"] = {"path": str(BASE / "verification-output-current-01.json"), "sha256": sha(BASE / "verification-output-current-01.json"), "result": replay["result"], "softmax": replay["softmax"], "ppo": replay["ppo"]}
audit["result"] = "全部來源與既有artifact雜湊一致；兩列修訂已核對原文與新算例；全部既有離線斷言通過。"
save_json(BASE / "audit-output-current-01.json", audit)
print(json.dumps({"result": audit["result"], "changed_rows": len(changed), "repo_sources": len(audit["repo_source_hashes"]), "inherited_artifacts": len(audit["inherited_artifact_hashes"]), "emoji": audit["emoji"], "utf8_exhaustive": audit["utf8_exhaustive"], "replayed_result": audit["replayed_result"]}, ensure_ascii=False))
