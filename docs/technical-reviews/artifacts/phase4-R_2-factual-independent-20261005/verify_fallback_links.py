"""Check the exact original locations of R.2's suggested rereading entry."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def section(path, identifier):
    raw = path.read_bytes()
    raw.decode("utf-8")
    hs = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i,h in enumerate(hs) if h[0].startswith(("## " + identifier + " ").encode()))
    end = hs[i + 1].start() if i + 1 < len(hs) else len(raw)
    return raw[hs[i].start():end], raw[:hs[i].start()].count(b"\n") + 1


def main():
    records = []
    for relative,identifier,expected_backread in (("course/chapters/0A.md", "A.8", "[A.7](0A.md#A.7)"), ("course/chapters/07.md", "7.11", "[7.1的答案標籤](#71-對話怎麼表示)")):
        path = ROOT / relative
        raw,first = section(path,identifier)
        text = raw.decode("utf-8")
        whole = path.read_bytes()
        snapshot = ART / "frozen-input" / relative
        assert snapshot.read_bytes() == whole
        start = text.rfind("<details>")
        assert start >= 0
        main_body,tail = text[:start],text[start:]
        assert expected_backread in main_body and expected_backread not in tail
        body_lines = main_body.splitlines()
        actual_paragraph = next(line for line in body_lines if expected_backread in line)
        line = first + body_lines.index(actual_paragraph)
        final_details_line = first + text[:start].count("\n")
        links = [{"label": a,"target": b} for a,b in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",tail)]
        records.append({"source": relative+"#"+identifier,"original_sha256": hashlib.sha256(whole).hexdigest(),"frozen_input": str(snapshot.relative_to(ROOT)),"section_sha256": hashlib.sha256(raw).hexdigest(),"backread_original_line": line,"backread_original_paragraph": actual_paragraph,"backread_target": expected_backread,"final_details_first_line": final_details_line,"final_details_links": links,"backread_in_body": True,"backread_in_final_details": False})
    r2,first=section(ROOT/"course/README.md","R.2")
    statement="跳讀遇到陌生概念時，使用小節末尾的回讀連結補一個必要步驟，再回到原問題。前置連結是找路工具；理解眼前例子需要的說明仍留在正文裡。"
    assert statement in r2.decode()
    result={"reviewer_task":"/root/phase4_factual_coordinator/factual_r_2","source":"course/README.md#R.2","source_sha256":hashlib.sha256(r2).hexdigest(),"statement":statement,"statement_original_line":first+r2.decode().splitlines().index(statement),"environment":{"python":sys.version,"python_executable":sys.executable,"device":"CPU; source text inspection only","platform":platform.platform()},"targets":records,"result":"The original needed rereading links are inside the lesson bodies and absent from the final supplements for A.8 and 7.11. R.2's footer-only routing advice does not describe these linked lessons accurately."}
    (ART/"fallback-link-check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"source_sha256":result["source_sha256"],"statement_original_line":result["statement_original_line"],"targets":[{"source":r["source"],"backread_original_line":r["backread_original_line"],"final_details_first_line":r["final_details_first_line"],"backread_in_body":r["backread_in_body"],"backread_in_final_details":r["backread_in_final_details"]} for r in records],"result":result["result"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
