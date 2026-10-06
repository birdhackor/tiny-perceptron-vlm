"""Same R.2 owner: bounded actual-current 7.11 navigation context callback."""
import ast
import difflib
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent
PRIOR=ART.parent
TASK="/root/phase4_factual_coordinator/factual_r_2"
PRIOR_REPORT_SHA="3ce18eabcbf32c023e4dbe3f476db3daa63e86edebdef8c23532ae9f7243baa7"


def sha(raw): return hashlib.sha256(raw).hexdigest()
def relative(path): return str(path.relative_to(ROOT))


def sections(raw):
    raw.decode("utf-8"); hs=list(re.finditer(rb"(?m)^## [^\r\n]+",raw)); out={}
    for i,h in enumerate(hs):
        m=re.match(rb"^## ([A-Z\d]+\.\d+) (.+)$",h[0]); assert m
        end=hs[i+1].start() if i+1<len(hs) else len(raw)
        out[m[1].decode()]={"body":raw[h.start():end],"line":raw[:h.start()].count(b"\n")+1,"title":m[2].decode()}
    return out


def original_code(path,names,constants=()):
    raw=path.read_bytes(); tree=ast.parse(raw); nodes=[]; locators=[]
    for node in tree.body:
        take=isinstance(node,ast.FunctionDef) and node.name in names
        if isinstance(node,ast.Assign): take |= any(isinstance(t,ast.Name) and t.id in constants for t in node.targets)
        if take:
            nodes.append(node); locators.append({"name":getattr(node,"name",next((t.id for t in getattr(node,"targets",[]) if isinstance(t,ast.Name)),"")),"lines":[node.lineno,node.end_lineno]})
    ns={"ROOT":ROOT,"Path":Path,"re":re}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),relative(path),"exec"),ns)
    return ns,{"original_path":relative(path),"sha256":sha(raw),"executed_ast_nodes":locators}


def main():
    prior_path=ART/"prior-R.2-report.opaque.json"; prior_bytes=prior_path.read_bytes()
    assert sha(prior_bytes)==PRIOR_REPORT_SHA
    own=json.loads(prior_bytes); assert own["reviewer_task"]==TASK and own["verdict"]=="pass"
    current_guide=(ROOT/"course/README.md").read_bytes(); r2=sections(current_guide)["R.2"]
    assert sha(r2["body"])==own["source_sha256"]=="2cdf530f696f655cb5a0afb405c477972a7b6b25cf31bcae6bc61a73335274f1"
    assert (ART/"current-R.2.md").read_bytes()==r2["body"]
    old07=(PRIOR/"frozen-input/course/chapters/07.md").read_bytes()
    current07=(ROOT/"course/chapters/07.md").read_bytes(); old=sections(old07); now=sections(current07)
    old11=old["7.11"]; current11=now["7.11"]
    assert (ART/"current-7.11.md").read_bytes()==current11["body"]
    assert old11["title"]==current11["title"]
    a=old11["body"].decode(); b=current11["body"].decode()
    old_lines=a.splitlines(keepends=True); new_lines=b.splitlines(keepends=True)
    changed=[i for i,(x,y) in enumerate(zip(old_lines,new_lines,strict=True)) if x!=y]
    assert len(changed)==1
    i=changed[0]; assert "見[7.12](#7.12)" in old_lines[i]
    assert "成績見[7.17](#7.17)" in new_lines[i] and "下一節[7.12](#7.12)先用四格材料" in new_lines[i]
    assert old_lines[i].replace("見[7.12](#7.12)","成績見[7.17](#7.17)；下一節[7.12](#7.12)先用四格材料說明題目家族的切分規則")==new_lines[i]
    diff="".join(difflib.unified_diff(old_lines,new_lines,fromfile="own-original-frozen-7.11",tofile="current-original-7.11"))
    (ART/"actual-7.11-diff.txt").write_text(diff,encoding="utf-8")
    fences=[]
    for j,(old_code,new_code) in enumerate(zip(re.findall(rb"```python\n(.*?)```",old11["body"],re.S),re.findall(rb"```python\n(.*?)```",current11["body"],re.S),strict=True),1):
        assert old_code==new_code; fences.append({"fence":j,"raw_code_sha256":sha(new_code),"byte_identical":True,"executed_in_callback":False})
    assert len(fences)==2 and b"![" not in r2["body"] and b"![" not in current11["body"]
    stable_sections=[]
    for lesson in ("7.17","7.18","7.19"):
        assert old[lesson]["body"]==now[lesson]["body"]
        stable_sections.append({"source":"course/chapters/07.md#"+lesson,"section_sha256":sha(now[lesson]["body"]),"unchanged_from_personally_inspected_original":True})
    stable_code=[]
    for p in ("scripts/build_course.py","scripts/export_course.py","scripts/check_technical_reviews.py"):
        old_code=(PRIOR/"frozen-input"/p).read_bytes(); new_code=(ROOT/p).read_bytes(); assert old_code==new_code
        stable_code.append({"original_path":p,"sha256":sha(new_code),"byte_identical_to_originally_inspected_contract":True})
    # Reuse only retained evidence relevant to the unchanged primary route map and
    # original converter contract. Every referenced permanent artifact is rehashed.
    reused=[]
    reuse_ids=("navigation-code","navigation-run","navigation-stdout","fallback-code","fallback-run","fallback-stdout","R2-reinspection","R2-reinspection-code","R2-reinspection-stdout")
    for identifier in reuse_ids:
        artifact=next(x for x in own["artifacts"] if x["id"]==identifier)
        p=ROOT/artifact["path"]; assert sha(p.read_bytes())==artifact["sha256"]
        reused.append({"artifact_id":identifier,"path":artifact["path"],"sha256":artifact["sha256"],"support_scope":"Historical original execution/source-position evidence only; unchanged primary route list and stable converter contract. This artifact does not describe the new 7.11 final-supplement sentence; callback receipt supplies its current context."})
    # Read only actual navigation fields, not author result notes or metrics.
    index=json.loads((ROOT/"course/lesson-index.json").read_bytes()); assert isinstance(index,list)
    rows={row["id"]:row for row in index}; targets={}; lesson_targets={}
    for row in index:
        path=(ROOT/row["source"]).resolve(); targets[path]="chapter-"+path.stem+".md"
        lesson_targets.setdefault(path,{})[row["id"]]=row["id"]+".md"
    contracts={}
    for lesson in ("7.1","7.11","7.12","7.17"):
        row=rows[lesson]; assert row["source"]=="course/chapters/07.md" and row["title"]==now[lesson]["title"]
        notebook=json.loads((ROOT/row["notebook"]).read_bytes()); assert notebook["metadata"]["lesson_id"]==lesson
        assert "".join(notebook["cells"][0]["source"]).strip()=="# "+lesson+" "+row["title"]
        contracts[lesson]={key:row[key] for key in ("id","title","source","notebook")}
    bc,bce=original_code(ROOT/"scripts/build_course.py",("lesson_anchor_aliases","notebook_reading_links"),("COURSE_URL","READING_PAGES"))
    ex,exe=original_code(ROOT/"scripts/export_course.py",("reading_markdown",),("COURSE_URL","REPOSITORY")); ex["lesson_anchor_aliases"]=bc["lesson_anchor_aliases"]
    location=[]
    split=b.rfind("<details>"); assert split>=0
    body=b[:split]; footer=b[split:]
    backreads=[(label,target) for label,target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",body) if target=="#71-對話怎麼表示"]
    assert len(backreads)==2
    for label,target in backreads:
        text="["+label+"]("+target+")"; observed=ex["reading_markdown"](text,ROOT/"course/chapters/07.md",targets,lesson_targets,"main").strip()
        assert observed=="["+label+"](7.1.md)"
        location.append({"input":text,"reading_target":"7.1.md","location":"body","unchanged_from_initial":True})
    for fragment in ("7.17","7.12"):
        matching=[(label,target) for label,target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",footer) if target=="#"+fragment]; assert len(matching)==1
        label,target=matching[0]; text="["+label+"]("+target+")"; observed=ex["reading_markdown"](text,ROOT/"course/chapters/07.md",targets,lesson_targets,"main").strip()
        assert observed=="["+label+"]("+fragment+".md)"
        assert bc["notebook_reading_links"](text,ROOT/"course/chapters/07.md")=="["+label+"]("+bc["COURSE_URL"]+fragment+".html)"
        location.append({"input":text,"reading_target":fragment+".md","notebook_reading_url":bc["COURSE_URL"]+fragment+".html","location":"final supplement"})
    r2_row=next(line for line in r2["body"].decode().splitlines() if "chapters/07.md#7.11" in line)
    mapped=ex["reading_markdown"](r2_row,ROOT/"course/README.md",targets,lesson_targets,"main").strip()
    assert "(7.11.md)" in mapped and "(7.17.md)" in mapped
    opening12="\n".join(now["7.12"]["body"].decode().splitlines()[:14])+"\n"
    assert "兩顏色×兩形狀，共四種組合" in opening12 and "沒有訓練模型或產生答題成績" in opening12
    (ART/"current-7.12-navigation-opening.md").write_text(opening12,encoding="utf-8")
    # Precise changed-context locator; save the actual current original section rather
    # than overwriting the frozen initial chapter or claiming old context is current.
    receipt={"reviewer_task":TASK,"date":"2026-10-06","source":"course/README.md#R.2","source_sha256":sha(r2["body"]),"primary_unchanged":True,"primary_first_line":r2["line"],"prior_opaque_report":{"path":relative(prior_path),"sha256":sha(prior_bytes),"issues_and_history_preserved_as_raw_report_bytes":True},"current_context":{"source":"course/chapters/07.md#7.11","raw_section_path":relative(ART/"current-7.11.md"),"source_sha256":sha(current11["body"]),"first_line":current11["line"],"changed_line":current11["line"]+i,"actual_changed_paragraph":new_lines[i].rstrip("\n"),"initial_source_sha256":sha(old11["body"]),"initial_whole_source_path":relative(PRIOR/"frozen-input/course/chapters/07.md"),"initial_whole_source_sha256":sha(old07)},"current_whole_inputs":[{"path":relative(ART/"current-input/course/README.md"),"sha256":sha(current_guide),"meaning":"whole guide frozen at callback time; not a continuing whole-guide version claim"},{"path":relative(ART/"current-input/course/chapters/07.md"),"sha256":sha(current07),"meaning":"whole chapter frozen at callback time; only necessary 7.11 and 7.12 navigation opening were personally read in this callback"}],"read_locators":["current full R.2","current full 7.11 (374–418), including its only changed paragraph at line "+str(current11["line"]+i),"current 7.12 navigation opening, 419–432; title and paper data-split table only","latest factual-reviewer-instructions.md full","exact latest clear-tutorial references/review-protocol.md full"],"r2_route_row":r2_row,"actual_current_row_conversion":mapped,"current_context_conversions":location,"current_selected_inventory":contracts,"checked_pointers":{"course/lesson-index.json":["/{selected 7.1,7.11,7.12,7.17 rows}/id","/title","/source","/notebook"],"selected_notebooks":["/metadata/lesson_id","/cells/0/source"]},"unchanged_context_code_fences":fences,"stable_original_code":stable_code,"stable_own_other_chapter7_targets":stable_sections,"reused_original_artifacts":reused,"actual_original_converter_execution":[bce,exe],"environment":{"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"device":"CPU source/navigation inspection only; no tensor or model calculation"},"scope_judgment":"The changed 7.11 final-supplement navigation separates the unchanged result location 7.17 from the 7.12 paper split example. R.2 still correctly introduces 7.11 as the dialogue demonstration and then routes to 7.17's two-stage explanation. Its corrected body-plus-supplement backread wording still agrees with the actual unchanged 7.1 body links. No R.2 concept, empirical result, numeric example or model capability claim changed.","reuse_limit":"Prior CPU artifacts are retained unchanged historical navigation/source-position evidence with exact SHA checks; they are not reruns and do not certify current 7.11 empirical results, current chapter-wide prose, model quality, or 7.12 science. New callback evidence certifies only the changed navigation context. No papers, training/model/data download, model inference, training, build, publication, reading-time or unrelated CPU check was performed.","verdict":"pass"}
    (ART/"context-callback-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"reviewer_task":TASK,"primary_source_sha256":receipt["source_sha256"],"context_7_11_sha256":receipt["current_context"]["source_sha256"],"prior_opaque_sha256":sha(prior_bytes),"changed_context_line":receipt["current_context"]["changed_line"],"scope_judgment":receipt["scope_judgment"],"verdict":"pass"},ensure_ascii=False))


if __name__=="__main__": main()
