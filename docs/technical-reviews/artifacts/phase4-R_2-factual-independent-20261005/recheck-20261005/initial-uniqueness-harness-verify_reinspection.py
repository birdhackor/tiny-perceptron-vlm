"""Original R.2 reviewer's bounded reinspection after one navigation sentence fix."""
import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
PRIOR = ART.parent
TASK = "/root/phase4_factual_coordinator/factual_r_2"
EXPECTED_PRIOR_REPORT = "b16b14bfad7c6826103cfb384b2f14a406f1f3a76168c05eb14cd394d2577b29"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(raw):
    raw.decode("utf-8")
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    out = {}
    for i,h in enumerate(headings):
        m=re.match(rb"^## ([A-Z\d]+\.\d+) (.+)$",h[0]); assert m
        end=headings[i+1].start() if i+1<len(headings) else len(raw)
        out[m[1].decode()]={"body":raw[h.start():end],"line":raw[:h.start()].count(b"\n")+1,"title":m[2].decode()}
    return out


def freeze(relative):
    raw=(ROOT/relative).read_bytes(); target=ART/"frozen-input"/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert target.read_bytes()==raw
    else: target.write_bytes(raw)
    assert sha(target.read_bytes())==sha((ROOT/relative).read_bytes())
    return {"original_path":relative,"snapshot":str(target.relative_to(ROOT)),"sha256":sha(raw),"bytes":len(raw)}


def original_code(relative,names,constants=()):
    raw=(ROOT/relative).read_bytes(); tree=ast.parse(raw); nodes=[]; locators=[]
    for n in tree.body:
        take=isinstance(n,ast.FunctionDef) and n.name in names
        if isinstance(n,ast.Assign): take |= any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)
        if take:
            nodes.append(n); locators.append({"name":getattr(n,"name",next((t.id for t in getattr(n,"targets",[]) if isinstance(t,ast.Name)),"")),"lines":[n.lineno,n.end_lineno]})
    ns={"ROOT":ROOT,"Path":Path,"re":re}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,"exec"),ns)
    return ns,{"original_path":relative,"sha256":sha(raw),"executed_ast_nodes":locators}


def main():
    prior_report=PRIOR/"initial-report.json"; prior_raw=prior_report.read_bytes()
    assert sha(prior_raw)==EXPECTED_PRIOR_REPORT
    own=json.loads(prior_raw); assert own["reviewer_task"]==TASK and own["verdict"]=="revise"
    prior_copy=ART/"prior-report.json"
    if prior_copy.exists(): assert prior_copy.read_bytes()==prior_raw
    else: prior_copy.write_bytes(prior_raw)
    current_guide=(ROOT/"course/README.md").read_bytes()
    current=sections(current_guide)["R.2"]; raw=current["body"]
    prior_section=(PRIOR/"section.md").read_bytes()
    assert sha(prior_section)==own["source_sha256"]
    old_sentence="跳讀遇到陌生概念時，使用小節末尾的回讀連結補一個必要步驟，再回到原問題。前置連結是找路工具；理解眼前例子需要的說明仍留在正文裡。"
    new_sentence="跳讀遇到陌生概念時，使用正文中的回讀連結與小節末尾的補充連結，補一個必要步驟，再回到原問題。前置連結是找路工具；理解眼前例子需要的說明仍留在正文裡。"
    assert old_sentence.encode() in prior_section and new_sentence.encode() in raw
    assert prior_section.replace(old_sentence.encode(),new_sentence.encode())==raw
    (ART/"section.md").write_bytes(raw)
    assert b"```" not in raw and not re.search(rb"!\[|<(?:img|svg)\b",raw)
    links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)",raw.decode())
    old_links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)",prior_section.decode())
    assert links==old_links and len(links)==20
    # JSON shape is checked before accessing only named original navigation fields.
    index=json.loads((ROOT/"course/lesson-index.json").read_bytes()); assert isinstance(index,list)
    contract=json.loads((ROOT/"course/lesson-contract.json").read_bytes()); assert isinstance(contract,dict)
    assert contract["schema_version"]==1 and isinstance(contract["lesson_ids"],list)
    indexed={x["id"]:x for x in index}; assert len(indexed)==len(index)
    assert set(indexed)==set(contract["lesson_ids"])
    needed={str((ROOT/"course"/t.split("#")[0]).resolve().relative_to(ROOT)) for _,t in links}
    source_sections={p:sections((ROOT/p).read_bytes()) for p in needed}
    targets={(ROOT/row["source"]).resolve():"chapter-"+Path(row["source"]).stem+".md" for row in index}
    lesson_targets={}
    for row in index: lesson_targets.setdefault((ROOT/row["source"]).resolve(),{})[row["id"]]=row["id"]+".md"
    bc,bce=original_code("scripts/build_course.py",("lesson_anchor_aliases","notebook_reading_links"),("COURSE_URL","READING_PAGES"))
    ex,exe=original_code("scripts/export_course.py",("reading_markdown",),("COURSE_URL","REPOSITORY"))
    ex["lesson_anchor_aliases"]=bc["lesson_anchor_aliases"]
    converted=ex["reading_markdown"](raw.decode(),ROOT/"course/README.md",targets,lesson_targets,"main")
    converted_links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)",converted); assert len(converted_links)==len(links)
    checked=[]
    for ((label,target),(new_label,new_target)) in zip(links,converted_links,strict=True):
        relative,_,fragment=target.partition("#"); path=(ROOT/"course"/relative).resolve(); p=str(path.relative_to(ROOT))
        assert path.is_file() and source_sections[p]
        if fragment:
            record=source_sections[p][fragment]; row=indexed[fragment]
            assert row["source"]==p and row["title"]==record["title"] and fragment in contract["lesson_ids"]
            notebook=json.loads((ROOT/row["notebook"]).read_bytes())
            assert notebook["metadata"]["lesson_id"]==fragment
            assert "".join(notebook["cells"][0]["source"]).strip()=="# "+fragment+" "+record["title"]
            expected=fragment+".md"; nb_page=fragment+".html"
        else:
            expected="chapter-"+path.stem+".md"; nb_page="chapter-"+path.stem+".html"
        assert label==new_label and expected==new_target
        assert bc["notebook_reading_links"]("["+label+"]("+target+")",ROOT/"course/README.md")=="["+label+"]("+bc["COURSE_URL"]+nb_page+")"
        checked.append({"label":label,"original_target":target,"reading_target":new_target})
    backreads=[]
    for relative,lesson,target,expected_page in (("course/chapters/0A.md","A.8","0A.md#A.7","A.7.md"),("course/chapters/07.md","7.11","#71-對話怎麼表示","7.1.md")):
        record=source_sections[relative][lesson]; body=record["body"].decode(); start=body.rfind("<details>"); assert start>=0
        before,footer=body[:start],body[start:]
        found=[(a,b) for a,b in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",before) if b==target]; assert len(found)==1
        assert target not in footer
        label,link=found[0]; text="["+label+"]("+link+")"
        observed=ex["reading_markdown"](text,ROOT/relative,targets,lesson_targets,"main").strip()
        assert observed=="["+label+"]("+expected_page+")"
        paragraphs=body.splitlines(); containing=next(x for x in paragraphs if text in x)
        backreads.append({"source":relative+"#"+lesson,"body_backread":text,"body_line":record["line"]+paragraphs.index(containing),"final_details_first_line":record["line"]+body[:start].count("\n"),"body_not_footer":True,"actual_generated_target":expected_page,"original_section_sha256":sha(record["body"])})
    # The modified sentence fits both exact original link positions, while scope-bearing
    # material and original contracts are unchanged since this reviewer's own first check.
    stable_scope=[]
    for relative,identifiers in (("course/chapters/19.md",("19.1","19.4","19.12")),("course/chapters/20.md",("20.1",))):
        old=(PRIOR/"frozen-input"/relative).read_bytes(); now=(ROOT/relative).read_bytes()
        old_sections=sections(old); now_sections=sections(now)
        for lesson in identifiers:
            assert old_sections[lesson]["body"]==now_sections[lesson]["body"]
            stable_scope.append({"source":relative+"#"+lesson,"sha256":sha(now_sections[lesson]["body"]),"unchanged_from_personally_inspected_initial_source":True})
    stable_contracts=[]
    for relative in ("docs/course-revision-20261005/outline.md","docs/course-revision-20261005/rewrite-contract.md","scripts/build_course.py","scripts/export_course.py"):
        old=(PRIOR/"frozen-input"/relative).read_bytes(); now=(ROOT/relative).read_bytes(); assert old==now
        stable_contracts.append({"original_path":relative,"sha256":sha(now),"unchanged_from_personally_inspected_initial_source":True})
    copies=[freeze(p) for p in sorted(needed | {"course/README.md","course/lesson-index.json","course/lesson-contract.json","scripts/build_course.py","scripts/export_course.py","docs/course-revision-20261005/outline.md","docs/course-revision-20261005/rewrite-contract.md"})]
    result={"reviewer_task":TASK,"source":"course/README.md#R.2","source_sha256":sha(raw),"section_first_line":current["line"],"newline_policy":"original UTF-8 bytes, no stripping or newline normalization","prior_report_sha256":sha(prior_raw),"prior_report_path":str(prior_copy.relative_to(ROOT)),"prior_source_sha256":sha(prior_section),"original_issue_id":"R2-footer-only-backread","original_issue_sentence":old_sentence,"current_sentence":new_sentence,"current_sentence_original_line":current["line"]+raw.decode().splitlines().index(new_sentence),"exact_single_sentence_replacement":True,"unchanged_links":checked,"backread_location_rechecks":backreads,"scope_material_unchanged":stable_scope,"original_contracts_unchanged":stable_contracts,"snapshots":copies,"original_code_executed":[bce,exe],"environment":{"python":sys.version,"python_executable":sys.executable,"device":"CPU; source and navigation checks only","platform":platform.platform()},"checked_pointers":{"course/lesson-index.json":["/{selected row}/id","/{selected row}/title","/{selected row}/source","/{selected row}/notebook"],"course/lesson-contract.json":["/schema_version","/lesson_ids"],"selected_target_notebooks":["/metadata/lesson_id","/cells/0/source"]},"result":"Original issue resolved: current R.2 names body rereading links and final supplementary links. Both exact original backreads are located in the body and convert to the correct earlier lesson page. All unchanged 20 route links still resolve through the original current converters; original finite-scope dependencies remain byte-identical. No build, model, training, publication or unrelated computation ran."}
    (ART/"reinspection.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ART/"current-converted-R.2.md").write_text(converted,encoding="utf-8")
    print(json.dumps({"reviewer_task":TASK,"source_sha256":sha(raw),"prior_source_sha256":sha(prior_section),"route_links":len(checked),"backread_rechecks":backreads,"result":result["result"]},ensure_ascii=False))


if __name__=="__main__": main()
