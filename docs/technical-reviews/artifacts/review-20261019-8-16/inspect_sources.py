"""讀取指定原始來源，只保存定位、雜湊與必要的短摘錄。"""

import ast
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "60", url]
    completed = subprocess.run(command, check=True, capture_output=True)
    raw = completed.stdout
    return raw, {
        "url": url, "command_argv": command, "exit_code": completed.returncode,
        "fetched_at": datetime.now(UTC).isoformat(),
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "full_content_saved": False,
    }


def excerpt(lines, start, end):
    return {"locator": f"lines {start + 1}-{end}", "text": "\n".join(lines[start:end])}


def main():
    records = []
    raw, paper = fetch("https://arxiv.org/pdf/2311.07911v1")
    assert raw.startswith(b"%PDF")
    paper.update(
        version="arXiv:2311.07911v1 (2023-11-15)",
        inspected="以 web PDF 原文讀取並核對 Table 1、IFEVAL METRICS、EVALUATION RESULTS；curl 另取同 URL 原檔雜湊。",
        locators=["Table 1: Detectable Format / JSON Format", "IFEVAL METRICS: strict 與 loose 的定義及誤判限制", "EVALUATION RESULTS: prompt-level 與 instruction-level 分母定義"],
        excerpts=[
            "The percentage of prompts that all verifiable instructions in each prompt are followed.",
            "Entire output should be wrapped in JSON format.",
        ],
        inspection_note="論文將逐指令與同題全部指令通過分開；loose 以轉換過的文字再驗證，並明說可能增加假陽性。其輸入是 response text，沒有生成 token 或 EOS 停止原因。",
    )
    records.append(paper)
    raw, ifeval = fetch("https://raw.githubusercontent.com/google-research/google-research/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/instruction_following_eval/instructions.py")
    code = raw.decode()
    tree = ast.parse(code)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JsonFormat")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "check_following")
    env = {"json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "JsonFormat.check_following", "exec"), env)
    cases = [
        ("markdown_json", '```json\n{"text":["今日休館","明日開放"]}\n```', True),
        ("extra_field", '{"text":["今日休館","明日開放"],"en":["Closed today"]}', True),
        ("non_list_value", '{"text":4}', True),
        ("json_scalar", '42', True),
        ("prose_prefix", '以下是結果：{"text":[]}', False),
    ]
    probes = []
    for name, value, expected in cases:
        observed = env["check_following"](None, value)
        assert observed == expected
        probes.append({"name": name, "value": value, "expected": expected, "observed": observed})
    ifeval.update(version="e49bbfe381c9c0e564b937f1c4e163a2273c65cc", excerpts=[excerpt(code.splitlines(), method.lineno - 1, method.end_lineno)], probes=probes,
                  inspection_note="直接讀 JsonFormat.check_following，輸入只有 value 字串；移除特定 Markdown 外框後只做 json.loads，不驗證 text 欄位、清單型別或 EOS。擷取原方法實跑五個反例核對其接受界線。")
    records.append(ifeval)
    raw, hf = fetch("https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py")
    lines = raw.decode().splitlines()
    terms = {"max_length (`int`": 3, "max_new_tokens (`int`": 2, "forced_eos_token_id (`int`": 3, "eos_token_id (`Union": 2}
    hf_excerpts = []
    for term, count in terms.items():
        start = next(i for i, line in enumerate(lines) if term in line)
        hf_excerpts.append(excerpt(lines, start, start + count))
    hf.update(version="Transformers v4.57.1", excerpts=hf_excerpts,
              inspection_note="GenerationConfig 分別定義長度上限、EOS ID，以及在達到 max_length 時強制輸出末 token 的 forced_eos_token_id；因此文字尾端或單獨看 EOS ID，均不足以推定模型自行結束。")
    records.append(hf)
    for name, selectors in [
        ("json", {"If *ensure_ascii* is true (the default)": 3, ".. function:: dumps(": 8, ".. function:: loads(": 10}),
        ("functions", {".. function:: all(": 9, ".. function:: any(": 9, ".. function:: isinstance(": 8, ".. function:: sorted(": 7}),
        ("stdtypes", {".. method:: str.find(": 10, ".. describe:: list(d)": 5, ".. describe:: iter(d)": 5}),
    ]:
        url = f"https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/library/{name}.rst"
        raw, record = fetch(url)
        lines = raw.decode().splitlines()
        excerpts = []
        for term, count in selectors.items():
            start = next(i for i, line in enumerate(lines) if term in line)
            excerpts.append(excerpt(lines, start, start + count))
        record.update(version="CPython v3.13.0 official documentation source; executed interpreter 3.13.16", excerpts=excerpts,
                      inspection_note="核對本節使用的標準函式定義；用當次 Python 執行另驗證實際行為。只保存與主張直接相關的小段文件。")
        records.append(record)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-16/inspect_sources.py",
        "environment": {"python": platform.python_version(), "device": "cpu", "curl": subprocess.run(["curl", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]},
        "records": records,
        "result": "所有原始來源讀取成功，IFEval 原方法的五個實跑結果均符合預期；第三方全文未存檔。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
