"""保留既有案例輸入，只調整已移除的圍欄函式介面後重跑探針。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main(args):
    repo, area = args.repo.resolve(), args.area.resolve()
    area.mkdir()
    art = repo / "docs/technical-reviews/artifacts/review-20261019-review-tools"
    helper = '\n    def guard_observation(text, position):\n        try:\n            return gate.fenced(text, position, "fixture.md")\n        except ValueError as error:\n            return {"rejected_not_plain": True, "error": str(error)}\n\n'
    names = ["author-patch-probe-current-03.py", "fence-admission-probe-current-04.py", "fence-semantics-probe-current-04.py"]
    rows = []
    for name in names:
        original = (art / name).read_bytes()
        code = original.decode()
        if "semantics" not in name:
            needle = 'gate.fenced(live, live.find("new-token"))'
            assert code.count(needle) == 1
            code = code.replace(needle, 'guard_observation(live, live.find("new-token"))')
            needle = '    original_chapter = unit.CHAPTER\n'
            assert code.count(needle) == 1
            code = code.replace(needle, helper + needle)
        else:
            start = code.index('        super_guard = gate.inside_fence(')
            end = code.index('        rows.append(', start)
            replacement = """        rejected = False
        try:
            block_observation = {"blocks": gate.fence_blocks(text, "fixture.md")}
            union = gate.fenced(text, position, "fixture.md")
        except ValueError as error:
            rejected = True
            block_observation = {"rejected_not_plain": True, "error": str(error)}
            union = None
        in_super = any(marker in block for block in blocks.blocks)
        in_common_fence = any(marker in block["content"] for block in fences)
"""
            code = code[:start] + replacement + code[end:]
            code = code.replace('{"superfences": super_guard, "commonmark": common_guard, "union": union}', '{"fence_blocks": block_observation, "fenced": union}')
            code = code.replace('"unsafe_union": (in_super or in_common_fence) and not union', '"unsafe_union": (in_super or in_common_fence) and not union and not rejected')
        target = area / name
        with target.open("x") as stream:
            stream.write(code)
        command = [sys.executable, "-B", str(target), "--repo", str(repo)]
        if "semantics" not in name:
            command += ["--area", str(area / (name.removesuffix(".py") + "-fixtures"))]
        command += ["--markdown-root", "/Users/yclin/.cache/uv/archive-v0/_s1u9QcvnBYGXI-2", "--pymdownx-root", "/Users/yclin/.cache/uv/archive-v0/4qFlT7bxJOvyO4qC"]
        if "semantics" in name:
            command += ["--commonmark-root", "/Users/yclin/.cache/uv/archive-v0/G9AZbu6IOXfcgErk", "--mdurl-root", "/Users/yclin/.cache/uv/archive-v0/-TXZPbBw8D3ICYU3QHDYY"]
        result = subprocess.run(command, cwd=repo, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
        rows.append({"original_script": name, "original_sha256": hashlib.sha256(original).hexdigest(),
                     "adapted_script": str(target), "adapted_sha256": hashlib.sha256(code.encode()).hexdigest(),
                     "changes": "只適配 fenced 的 where 引數、記錄不單純檔案的 ValueError；移除已不存在的兩個子掃描器觀察。案例輸入、補丁與 fixture 准入流程不變。",
                     "command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(json.dumps({"probes": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    main(parser.parse_args())
